import re
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from arxiv_utils import ArXiv
from get_stock_info import get_sentiment
from core import convert_audio_file_to_format


OUTPUT_FORMAT = "mp3"

file_matcher = re.compile(r"Correcting container of \"(.*)\"")
file_matcher2 = re.compile(r"\[download\] Destination: (.*)")
file_matcher3 = re.compile(r"\[download\] (.*) has already been downloaded")


@dataclass
class BotResponse:
    kind: str
    text: Optional[str] = None
    parse_mode: Optional[str] = None
    file_path: Optional[str] = None
    cleanup_path: bool = False


@dataclass
class BotResult:
    responses: List[BotResponse]
    research_query: Optional[str] = None
    transcribed_text: Optional[str] = None


class BotCore:
    def __init__(self, llm_service, deep_research_runner: Callable[[str], str]):
        self.llm_service = llm_service
        self.deep_research_runner = deep_research_runner

    def ensure_state(self, state: dict) -> None:
        state.setdefault("chat_history", [])
        state.setdefault("writer_mode", False)
        state.setdefault("use_context_summary", True)
        state.setdefault("history", [])

    def append_chat_history(self, state: dict, text: str, reply_text: Optional[str]) -> None:
        history = state.setdefault("chat_history", [])
        if reply_text:
            history.append(reply_text)
        history.append(text)

    async def build_research_query(
        self,
        state: dict,
        current_text: str,
        reply_text: Optional[str],
    ) -> str:
        history = state.setdefault("chat_history", [])
        past_snippets = history[-5:]
        context_summary = ""
        if state.get("use_context_summary", True):
            context_summary = await self.llm_service.summarize_past_discussions(past_snippets)
        self.append_chat_history(state, current_text, reply_text)
        if context_summary:
            return (
                "Context summary of prior discussions:\n"
                f"{context_summary}\n\n"
                "Current query:\n"
                f"{current_text}"
            )
        return current_text

    def toggle_writer(self, state: dict) -> bool:
        state["writer_mode"] = not state.get("writer_mode", False)
        return state["writer_mode"]

    def toggle_context_summary(self, state: dict) -> bool:
        state["use_context_summary"] = not state.get("use_context_summary", True)
        return state["use_context_summary"]

    async def handle_text(
        self,
        state: dict,
        text: str,
        reply_text: Optional[str] = None,
        reply_chain: Optional[List[str]] = None,
    ) -> List[BotResponse]:
        self.append_chat_history(state, text, reply_text)

        if text.startswith("https://arxiv.org/"):
            paper = ArXiv(text)
            summary = await self.llm_service.summarize_paper_sections(paper)
            paper.summary = summary
            return [
                BotResponse(kind="text", text=msg, parse_mode="HTML")
                for msg in paper.to_message()
            ]

        if text.startswith("https://www.youtube.com/watch?") or text.startswith("https://youtu.be/"):
            from subprocess import check_output

            output = check_output(f"yt-dlp --cookies ../youtube_cookie.txt -f 140 {text}", shell=True).decode("utf-8")
            output_file = ""
            for line in output.split("\n"):
                m = file_matcher.search(line) or file_matcher2.search(line) or file_matcher3.search(line)
                if m:
                    output_file = m.group(1).strip()
                    break
            if output_file:
                return [BotResponse(kind="audio", file_path=output_file, cleanup_path=True)]
            return [BotResponse(kind="text", text="Failed to extract audio from youtube link.")]

        if text.startswith("a:"):
            _, keywords = text.split(":", 1)
            papers = ArXiv.search_arxiv(keywords.split())
            responses = []
            for paper in papers:
                for msg in paper.to_message():
                    responses.append(BotResponse(kind="text", text=msg, parse_mode="HTML"))
            return responses

        if text == "bs":
            chain = reply_chain or []
            if not chain:
                return [BotResponse(kind="text", text="No reply chain found for brainstorming.")]
            keywords = await self.llm_service.summarize_keywords(chain)
            papers = ArXiv.search_arxiv(keywords)
            responses = [
                BotResponse(kind="text", text=f"Keywords: {keywords}. Find {len(papers)} papers")
            ]
            reference_idea = " ".join(chain)
            for paper in papers:
                paper.summary = await self.llm_service.summarize_paper_sections(
                    paper,
                    reference_idea=reference_idea,
                )
                for msg in paper.to_message():
                    responses.append(BotResponse(kind="text", text=msg, parse_mode="HTML"))
            return responses

        if text.startswith("search"):
            item = text.split(" ", 1)[1].strip()
            _, overall_output = get_sentiment(item)
            overall_output = overall_output.replace("[", "<b>").replace("]", "</b>")
            return [BotResponse(kind="text", text=overall_output, parse_mode="HTML")]

        return [BotResponse(kind="text", text="I don't understand")]

    async def handle_voice(
        self,
        state: dict,
        voice_bytes: bytes,
        reply_text: Optional[str] = None,
        log_research_query: Optional[Callable[[str], None]] = None,
        message_date=None,
    ) -> BotResult:
        with tempfile.NamedTemporaryFile("wb+", suffix=".ogg") as temp_audio_file:
            temp_audio_file.write(voice_bytes)
            temp_audio_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=f".{OUTPUT_FORMAT}") as temp_output_file:
                convert_audio_file_to_format(temp_audio_file.name, temp_output_file.name, OUTPUT_FORMAT)
                transcribed_text = await self.llm_service.transcribe_audio(temp_output_file.name)

        responses = [
            BotResponse(kind="text", text="Transcribed text:"),
            BotResponse(kind="text", text=transcribed_text),
        ]

        research_query = await self.build_research_query(state, transcribed_text, reply_text)
        if log_research_query:
            log_research_query(research_query)
        research_answer = self.deep_research_runner(research_query)

        if not research_answer:
            responses.append(BotResponse(kind="text", text="Deep research returned no answer."))
        else:
            responses.append(BotResponse(kind="text", text="Starting deep research..."))
            responses.append(BotResponse(kind="text", text=research_answer))
            self.append_chat_history(state, research_answer, reply_text=None)

        if state.get("writer_mode", False):
            result_obj = await self.llm_service.preprocess_text(transcribed_text)
            model_family = "gemini-2.5-flash" if result_obj.get("tag") == "聊天" else "gemini-2.5-pro-preview-05-06"
            result_obj["model"] = model_family
            result_obj["transcribed"] = transcribed_text
            paraphrased_text = await self.llm_service.paraphrase_text(result_obj["content"], model_family)
            result_obj["paraphrased"] = paraphrased_text
            result_obj["date"] = message_date
            state.setdefault("history", []).append(result_obj)
            responses.append(BotResponse(kind="text", text=f"Paraphrased using {model_family}:"))
            responses.append(BotResponse(kind="text", text=paraphrased_text))

        return BotResult(
            responses=responses,
            research_query=research_query,
            transcribed_text=transcribed_text,
        )
