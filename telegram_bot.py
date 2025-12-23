from typing import List
import os
import sys
import tempfile
import json
from telegram import Update, BotCommand, Bot
from telegram.constants import ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Application, PicklePersistence
import telegram.ext.filters as filters
from subprocess import check_output, run, CalledProcessError
import re

import core
from core import paraphrase_text, convert_audio_file_to_format, preprocess_text
import requests
from bs4 import BeautifulSoup

from llm_summary import ModelInterface

LLM_UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "llm_utils")
if LLM_UTILS_DIR not in sys.path:
    sys.path.insert(0, LLM_UTILS_DIR)

from llm_util import transcribe_audio_gemini
from arxiv_utils import ArXiv
from get_stock_info import get_sentiment

OUTPUT_FORMAT = "mp3"
TELEGRAM_MESSAGE_LIMIT = 4096

DEEP_RESEARCH_DIR = os.environ.get("DEEP_RESEARCH_DIR", "/home/yuandong/Tongyi/inference")
DEEP_RESEARCH_ENV_FILE = os.environ.get("DEEP_RESEARCH_ENV_FILE", "/home/yuandong/Tongyi/.env")
DEEP_RESEARCH_MODEL = os.environ.get("DEEP_RESEARCH_MODEL", "")

telegram_api_token = os.environ.get('TELEGRAM_BOT_TOKEN')
print(f'Bot token: {telegram_api_token}')

telegram_allow_user_name = os.environ.get("TELEGRAM_ALLOW_USER")
print(f"Allow user name: {telegram_allow_user_name}")

writer_mode = False
print(f"Writer's mode: {writer_mode}")

model = ModelInterface()

def load_env_file(path: str, base_env: dict) -> dict:
    if not path or not os.path.exists(path):
        return base_env
    env = dict(base_env)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            env.setdefault(key, value)
    return env

def split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> List[str]:
    if not text:
        return [""]
    chunks = []
    current = []
    current_len = 0
    for paragraph in text.splitlines(keepends=True):
        if current_len + len(paragraph) > limit and current:
            chunks.append("".join(current).rstrip())
            current = []
            current_len = 0
        if len(paragraph) > limit:
            for i in range(0, len(paragraph), limit):
                chunks.append(paragraph[i:i + limit].rstrip())
            continue
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        chunks.append("".join(current).rstrip())
    return chunks

def run_deep_research(query: str) -> str:
    run_dir = DEEP_RESEARCH_DIR
    script_path = os.path.join(run_dir, "run_deep_research.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Deep research script not found at {script_path}")

    with tempfile.NamedTemporaryFile(prefix="deep_research_", suffix=".json", delete=False) as output_file:
        output_path = output_file.name

    cmd = [
        "python",
        "-u",
        script_path,
        "--query",
        query,
        "--output_file",
        output_path,
    ]
    if DEEP_RESEARCH_MODEL:
        cmd.extend(["--model", DEEP_RESEARCH_MODEL])

    env = load_env_file(DEEP_RESEARCH_ENV_FILE, os.environ)
    result = run(cmd, capture_output=True, text=True, cwd=run_dir, env=env, check=False)
    if result.returncode != 0:
        raise CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

    with open(output_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("final_answer") or payload.get("prediction") or ""

async def start(update: Update, context: CallbackContext):
    '''
        Start the bot
    '''
    global commands
    bot = Bot(telegram_api_token)
    curr_commands = await bot.get_my_commands()
    await update.message.reply_text(curr_commands)

    await bot.set_my_commands([BotCommand(command=f.__name__, description=f.__doc__) for f in commands])

    await update.message.reply_text('Send me a voice message, and I will transcribe it for you. Note I am not a QA bot, and will not answer your questions. I will only listen to you and transcribe your voice message, with paraphrasing from GPT-4. Type /help for more information.')

async def help(update: Update, context: CallbackContext):
    '''
        Get help message
    '''
    await update.message.reply_text("""*YaGe Voice Note Taker Bot*

*Usage*: Send me a voice message, and I will transcribe it for you\. Note I am not a QA bot, and will not answer your questions\. I will only listen to you and transcribe your voice message, with paraphrasing from GPT\-4\.

*Data and privacy*: I log your transcriptions and paraphrased texts, to support a future service of sending summary of your voice messages\. I will not share your data with any third party\. I will not use your data for any purposes other than to provide you with a better service\. You can always check what data are logged by sending /data command, and clear your data \(on our end\) by sending /clear command\.

*Commands*: 
/help: Display this help message\.
/data: Display any information we had about you from our end\.
/clear: Clear any information we had about you from our end\.""", parse_mode='MarkdownV2')

async def data(update: Update, context: CallbackContext):
    """
    Display any information we had about the user from our end.
    """
    chat_id = context._chat_id
    user_id = context._user_id
    member = await context.bot.get_chat_member(chat_id, user_id)
    user_full_name = member.user.full_name
    print(f'[{user_full_name}] /data')
    to_send = str(context.user_data)
    if len(to_send) > 4096:
        await update.message.reply_text(f"Your data is too long to be displayed. It contains {len(context.user_data['history'])} entries. The last message is {context.user_data['history'][-1]}. It records across the time period from {context.user_data['history'][0]['date']} to {context.user_data['history'][-1]['date']}.")
    else:
        await update.message.reply_text(to_send)

async def clear(update: Update, context: CallbackContext):
    """
    Clear any information we had about the user from our end.
    """
    chat_id = context._chat_id
    user_id = context._user_id
    member = await context.bot.get_chat_member(chat_id, user_id)
    user_full_name = member.user.full_name
    print(f'[{user_full_name}] /clear')
    context.user_data.clear()
    await update.message.reply_text("Your data has been cleared.")

async def toggle_writer(update: Update, context: CallbackContext):
    """
    Switch writer's mode.
    """
    user_full_name = await check_auth(update, context)
    if user_full_name is None:
        return 

    global writer_mode

    writer_mode = not writer_mode
    await update.message.reply_text(f"Writer's mode is set to be {writer_mode}")

# TODO: send out daily summaries to users.
async def check_auth(update: Update, context: CallbackContext):
    chat_id = context._chat_id
    user_id = context._user_id
    member = await context.bot.get_chat_member(chat_id, user_id)
    user_full_name = member.user.full_name

    if user_full_name != telegram_allow_user_name:
        # not allowed.  
        await update.message.reply_text(f"You ({user_full_name}) is not in the allowed user list.")
        return None

    # We need to log the user info and histories in the user_data so we can send out daily summaries.
    # Check the help message for more details.
    if 'user_full_name' not in context.user_data:
        context.user_data['user_full_name'] = user_full_name
    if 'user_id' not in context.user_data:
        context.user_data['user_id'] = user_id
    if 'history' not in context.user_data:
        context.user_data['history'] = []

    return user_full_name

file_matcher = re.compile(r"Correcting container of \"(.*)\"") 
file_matcher2 = re.compile(r"\[download\] Destination: (.*)")
file_matcher3 = re.compile(r"\[download\] (.*) has already been downloaded")

async def send_papers(update: Update, all_papers : List[ArXiv], reply_to_message_id=None):
    for paper in all_papers:
        for msg in paper.to_message(): 
            # reply to the previous message
            # get message id of the previous message
            message = await update.message.reply_text(msg, parse_mode=ParseMode.HTML, reply_to_message_id=reply_to_message_id)
            reply_to_message_id = message.message_id


async def handle_text_message(update: Update, context: CallbackContext):
    user_full_name = await check_auth(update, context)
    if user_full_name is None:
        return 

    text = update.message.text 
    msg_id = update.message.message_id

    if text.startswith("https://arxiv.org/"):
        # reply = f"Receive arXiv: {text}"
        # print(f'[{user_full_name}] {reply}')
        # await update.message.reply_text(reply)
        paper = ArXiv(text)
        summary = model.get_summary(paper)
        paper.summary = summary
        await send_papers(update, [paper], reply_to_message_id=msg_id)
        
    elif text.startswith("https://www.youtube.com/watch?") or text.startswith("https://youtu.be/"):
        # convert youtube to music and output
        print(f"[{user_full_name}] {text}")
        message = await update.message.reply_text(f"Converting youtube link to m4a file..", reply_to_message_id=msg_id)
        msg_id = message.message_id
        output = check_output(f"yt-dlp --cookies ../youtube_cookie.txt -f 140 {text}", shell=True).decode("utf-8")
        print(output)
        for line in output.split("\n"):
            m = file_matcher.search(line)
            if m:
                output_file = m.group(1).strip()
                break

            m = file_matcher2.search(line)
            if m:
                output_file = m.group(1).strip()
                break

            m = file_matcher3.search(line)
            if m:
                output_file = m.group(1).strip()
                break

        print("Extracted: \"" + output_file + "\"")

        await update.message.reply_audio(open(output_file, "rb"), reply_to_message_id=msg_id)
        # Delete the audio to save space. Telegram already save the audio.
        os.remove(output_file)

    elif text.startswith("a:"):
        _, keywords = text.split(":", 1)
        papers = ArXiv.search_arxiv(keywords.split())
        await send_papers(update, papers, reply_to_message_id=msg_id)

    elif text == "bs":
        # Start the brainstorming process
        # Backward trace the messages that the current messages are replying to. 
        previous_message = update.message.reply_to_message
        backward_chain = []
        while previous_message:
            backward_chain.append(previous_message["text"])
            previous_message = previous_message.reply_to_message

        # Then send the messages to LLM API to start the brainstorming process.
        backward_chain = backward_chain[::-1]
        keywords = model.summarize_keywords(backward_chain)
        papers = ArXiv.search_arxiv(keywords)
        message = await update.message.reply_text(f"Keywords: {keywords}. Find {len(papers)} papers", reply_to_message_id=msg_id)
        msg_id = message.message_id

        reference_idea = " ".join(backward_chain)
        for paper in papers:
            # Extract their summary
            paper.summary = model.get_summary(paper, reference_idea=reference_idea)
            await send_papers(update, [paper], reply_to_message_id=msg_id)

    elif text.startswith("search"):
        # search twitter
        item = text.split(" ", 1)[1].strip()
        overall_sentiment, overall_output = get_sentiment(item)
        overall_output = overall_output.replace("[", "<b>").replace("]", "</b>")
        await update.message.reply_text(overall_output, parse_mode=ParseMode.HTML, reply_to_message_id=msg_id)
    else:
        await update.message.reply_text("I don't understand", reply_to_message_id=msg_id)

async def warn_if_not_voice_message(update: Update, context: CallbackContext):
    if not update.message.voice:
        await update.message.reply_text("Please send me a voice message. I will transcribe it and paraphrase for you.")

async def transcribe_voice_message(update: Update, context: CallbackContext):
    user_full_name = await check_auth(update, context)
    if user_full_name is None:
        return 
    
    file_id = update.message.voice.file_id
    voice_file = await context.bot.get_file(file_id)
    
    # Download the voice message
    voice_data = await voice_file.download_as_bytearray()
    
    # Call the Whisper ASR API
    with tempfile.NamedTemporaryFile('wb+', suffix=f'.ogg') as temp_audio_file:
        temp_audio_file.write(voice_data)
        temp_audio_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix=f'.{OUTPUT_FORMAT}') as temp_output_file:
            output_path = temp_output_file.name
            convert_audio_file_to_format(temp_audio_file.name, output_path, OUTPUT_FORMAT)
            transcribed_text = transcribe_audio_gemini(output_path)
    print(f'[{user_full_name}] {transcribed_text}')
    await update.message.reply_text("Transcribed text:")
    await update.message.reply_text(transcribed_text)

    msg_id = update.message.message_id
    await update.message.reply_text("Starting deep research...", reply_to_message_id=msg_id)
    try:
        research_answer = run_deep_research(transcribed_text)
    except Exception as exc:
        await update.message.reply_text(f"Deep research failed: {exc}", reply_to_message_id=msg_id)
        return

    if not research_answer:
        await update.message.reply_text("Deep research returned no answer.", reply_to_message_id=msg_id)
    else:
        for chunk in split_for_telegram(research_answer):
            await update.message.reply_text(chunk, reply_to_message_id=msg_id)

    global writer_mode

    if writer_mode:
        # output a json list with content and tag
        preprocessed_text = preprocess_text(transcribed_text)
        print(f'[{user_full_name}] {preprocessed_text}')
        result_obj = json.loads(preprocessed_text)

        model = 'gpt-3.5-turbo' if result_obj['tag'] == '聊天' else 'gpt-4'
        result_obj['model'] = model
        result_obj['transcribed'] = transcribed_text
        print(f'[{user_full_name}] {result_obj}')
        paraphrased_text = paraphrase_text(result_obj['content'], model)
        result_obj['paraphrased'] = paraphrased_text
        result_obj['date'] = update.message.date
        print(f'[{user_full_name}] {paraphrased_text}')
        context.user_data['history'].append(result_obj)
        await update.message.reply_text(f"Paraphrased using {model.upper()}:")
        await update.message.reply_text(paraphrased_text)

commands = [start, help, clear, data, toggle_writer]

def main():
    persistence = PicklePersistence(filepath="gpt_archive.pickle")
    application = Application.builder().token(telegram_api_token).persistence(persistence).build()

    # on different commands - answer in Telegram
    [ application.add_handler(CommandHandler(f.__name__, f)) for f in commands ]

    # on non command i.e message
    application.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, transcribe_voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(~filters.VOICE & ~filters.TEXT & ~filters.COMMAND, warn_if_not_voice_message))

    # Run the bot until the user presses Ctrl-C
    print('Bot is running...')
    application.run_polling()

if __name__ == '__main__':
    main()
