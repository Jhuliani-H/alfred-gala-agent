"""Extra tools Alfred needs for the GAIA benchmark (Unit 4 of the Agents Course).

The gala tools cover guests, weather and Hub stats. GAIA questions instead need
Alfred to read web pages, YouTube captions and attached files.
"""

import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from smolagents import Tool
from youtube_transcript_api import YouTubeTranscriptApi

# Wikipedia and many other sites reject requests without a browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AlfredGAIA/1.0"}


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Downloads a web page and returns its text as markdown (tables included). "
        "Pages are long, so pass `focus` with a few keywords to get only the parts "
        "of the page that mention them."
    )
    inputs = {
        "url": {"type": "string", "description": "The full URL of the page to read."},
        "focus": {
            "type": "string",
            "description": "Optional keywords. Only the page sections that mention them are returned.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, max_chars: int = 6000, **kwargs):
        super().__init__(**kwargs)
        # ~1.5k tokens: keeps several page reads inside a small local model's context.
        self.max_chars = max_chars

    def forward(self, url: str, focus: str | None = None) -> str:
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Could not open {url}: {e}"

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "svg"]):
            tag.decompose()
        # Lists inside table cells would break each table row over several lines.
        for cell_list in soup.select("td ul, td ol, th ul, th ol"):
            cell_list.replace_with("; ".join(li.get_text(" ", strip=True) for li in cell_list.find_all("li")))
        text = markdownify(str(soup), heading_style="ATX", strip=["img"])
        # Wikipedia noise that wastes a small model's context: [[3]](#cite_note-...)
        # citation markers, [[edit](...)] section links and the "tooltip" titles on every link.
        text = re.sub(r"\[\[[^\]]*\]\]\(#cite_note[^)]*\)|\[\[edit\]\([^)]*\)\]", "", text)
        text = re.sub(r'\]\((\S+) "[^"]*"\)', r"](\1)", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

        if focus:
            text = self._focus(text, focus)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n...[page truncated; use `focus` to read other parts]"
        return text

    def _focus(self, text: str, focus: str) -> str:
        """Keep the sections that best match the focus keywords, in page order.

        Whole sections (heading plus body, plus their subsections) are kept so
        a table stays together with its heading; a keyword in the heading counts extra.
        """
        keywords = {w for w in re.findall(r"\w+", focus.lower()) if len(w) > 2}
        chunks = re.split(r"\n(?=#{1,6} )", text)
        if len(chunks) < 3:  # page without headings: fall back to paragraphs
            chunks = text.split("\n\n")
        levels = [len(c) - len(c.lstrip("#")) for c in chunks]  # 0 = not a heading

        def score(chunk: str) -> int:
            heading, _, body = chunk.partition("\n")
            return sum(3 * (k in heading.lower()) + (k in body.lower()) for k in keywords)

        best = sorted(((score(c), i) for i, c in enumerate(chunks) if score(c) > 0), reverse=True)
        keep, size = set(), 0
        for _, i in best:
            if size > self.max_chars:
                break
            # e.g. "## Discography" brings along "### Studio albums" and its table
            end = i + 1
            while levels[i] and end < len(chunks) and levels[end] > levels[i]:
                end += 1
            for j in range(i, end):
                if j not in keep:
                    keep.add(j)
                    size += len(chunks[j])
        if not keep:
            return f"No part of the page mentions '{focus}'. Beginning of the page:\n\n{text}"
        return "\n\n".join(chunks[i] for i in sorted(keep))


class YouTubeTranscriptTool(Tool):
    name = "youtube_transcript"
    description = (
        "Returns the spoken words (captions) of a YouTube video. "
        "It only gives the dialogue: it cannot see what is shown in the video."
    )
    inputs = {"url": {"type": "string", "description": "The YouTube video URL."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
        if not match:
            return f"Could not find a YouTube video id in {url}."
        try:
            transcript = YouTubeTranscriptApi().fetch(match.group(1))
        except Exception as e:
            return f"No transcript available ({type(e).__name__}: {e})."
        return " ".join(snippet.text for snippet in transcript)


def read_attachment(path: Path, max_chars: int = 8000) -> str:
    """Turn a GAIA attachment into text Alfred can read inside his prompt.

    Doing this up front (rather than as a tool) is more reliable with small
    models, which often forget to open the file at all.
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        text = "\n\n".join(f"Sheet '{name}':\n{df.to_csv(index=False)}" for name, df in sheets.items())
    elif suffix == ".csv":
        text = pd.read_csv(path).to_csv(index=False)
    elif suffix in {".mp3", ".wav", ".m4a", ".flac"}:
        text = "Transcript of the audio:\n" + transcribe_audio(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "(This is an image. Alfred's language model cannot see images.)"
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[file truncated]"
    return text


_whisper_model = None


def transcribe_audio(path: Path) -> str:
    """Speech to text with faster-whisper, on the CPU so the GPU stays free for the LLM."""
    global _whisper_model
    from faster_whisper import WhisperModel  # imported lazily: only audio questions need it

    if _whisper_model is None:
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = _whisper_model.transcribe(str(path))
    return " ".join(segment.text.strip() for segment in segments)
