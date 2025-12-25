from typing import List
import os
import tempfile
import json
from telegram import Update, BotCommand, Bot
from telegram.ext import (
    CommandHandler,
    MessageHandler,
    CallbackContext,
    Application,
    PicklePersistence,
    PersistenceInput,
)
import telegram.ext.filters as filters
from subprocess import run, CalledProcessError

from bot_core import BotCore
from llm_service import LLMService
TELEGRAM_MESSAGE_LIMIT = 4096

DEEP_RESEARCH_DIR = os.environ.get("DEEP_RESEARCH_DIR", "/home/yuandong/Tongyi/inference")
DEEP_RESEARCH_ENV_FILE = os.environ.get("DEEP_RESEARCH_ENV_FILE", "/home/yuandong/Tongyi/.env")
DEEP_RESEARCH_MODEL = os.environ.get("DEEP_RESEARCH_MODEL", "")

telegram_api_token = os.environ.get('TELEGRAM_BOT_TOKEN')
print(f'Bot token: {telegram_api_token}')

telegram_allow_user_name = os.environ.get("TELEGRAM_ALLOW_USER")
print(f"Allow user name: {telegram_allow_user_name}")


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

llm_service = LLMService(default_model="gemini-2.5-flash", use_cache=True)
bot_core = BotCore(llm_service=llm_service, deep_research_runner=run_deep_research)


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
/clear: Clear any information we had about you from our end\.
/toggle_writer: Toggle writer's mode\. 
/toggle_context_summary: Toggle context summary for deep research\.

""", parse_mode='MarkdownV2')

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
    writer_mode = bot_core.toggle_writer(context.user_data)
    await update.message.reply_text(f"Writer's mode is set to be {writer_mode}")

async def toggle_context_summary(update: Update, context: CallbackContext):
    """
    Toggle context summary.
    """
    user_full_name = await check_auth(update, context)
    if user_full_name is None:
        return

    use_context_summary = bot_core.toggle_context_summary(context.user_data)
    await update.message.reply_text(f"Context summary is set to be {use_context_summary}")

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
    bot_core.ensure_state(context.user_data)

    return user_full_name

def build_reply_chain(update: Update) -> List[str]:
    previous_message = update.message.reply_to_message
    backward_chain = []
    while previous_message:
        if previous_message.text:
            backward_chain.append(previous_message.text)
        previous_message = previous_message.reply_to_message
    return backward_chain[::-1]


async def handle_text_message(update: Update, context: CallbackContext):
    user_full_name = await check_auth(update, context)
    if user_full_name is None:
        return 

    text = update.message.text
    msg_id = update.message.message_id
    reply_text = update.message.reply_to_message.text if update.message.reply_to_message else None
    reply_chain = build_reply_chain(update) if text == "bs" else None
    responses = await bot_core.handle_text(context.user_data, text, reply_text=reply_text, reply_chain=reply_chain)
    for response in responses:
        if response.kind == "audio" and response.file_path:
            await update.message.reply_audio(open(response.file_path, "rb"), reply_to_message_id=msg_id)
            if response.cleanup_path:
                os.remove(response.file_path)
            continue
        if response.kind == "text" and response.text is not None:
            for chunk in split_for_telegram(response.text):
                await update.message.reply_text(
                    chunk,
                    parse_mode=response.parse_mode,
                    reply_to_message_id=msg_id,
                )

async def warn_if_not_voice_message(update: Update, context: CallbackContext):
    if not update.message.voice:
        await update.message.reply_text("Please send me a voice message. I will transcribe it and paraphrase for you.")

async def transcribe_voice_message(update: Update, context: CallbackContext):
    user_full_name = await check_auth(update, context)
    if user_full_name is None:
        return 
    
    file_id = update.message.voice.file_id
    voice_file = await context.bot.get_file(file_id)
    voice_data = await voice_file.download_as_bytearray()

    msg_id = update.message.message_id
    reply_text = update.message.reply_to_message.text if update.message.reply_to_message else None
    try:
        result = await bot_core.handle_voice(
            context.user_data,
            voice_data,
            reply_text=reply_text,
            log_research_query=lambda query: print(
                f'[{user_full_name}] Deep research query:\n{query}'
            ),
            message_date=update.message.date,
        )
    except Exception as exc:
        await update.message.reply_text(f"Deep research failed: {exc}", reply_to_message_id=msg_id)
        return

    if result.transcribed_text:
        print(f'[{user_full_name}] {result.transcribed_text}')

    for response in result.responses:
        if response.kind == "text" and response.text is not None:
            for chunk in split_for_telegram(response.text):
                await update.message.reply_text(chunk, reply_to_message_id=msg_id)

commands = [start, help, clear, data, toggle_writer, toggle_context_summary]

def main():
    persistence = PicklePersistence(
        filepath="gpt_archive.pickle",
        store_data=PersistenceInput(user_data=True, chat_data=True, bot_data=False),
    )
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
