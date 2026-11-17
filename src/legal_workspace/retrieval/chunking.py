import re
from dataclasses import dataclass
from threading import local

import snowballstemmer

CHUNKING_VERSION = "legal-spans-v1:900:120"
HEADING = re.compile(
    r"^\s*((?:(?:статья|глава)\s+\d+(?:\.\d+)*[.]?|"
    r"(?:раздел|часть)\s+(?:[IVXLC]+|\d+)[.]?)[^\n]{0,160})",
    re.I | re.M,
)
WORD = re.compile(r"[\w]+", re.UNICODE)
_THREAD = local()


def normalize(text: str) -> str:
    text = re.sub(r"\bст\.\s*(?=\d)", "статья ", text, flags=re.I)
    words = WORD.findall(text.lower().replace("ё", "е"))
    # Stemming retains numeric article references; a new normalization requires a rebuild.
    if not hasattr(_THREAD, "stemmer"):
        _THREAD.stemmer = snowballstemmer.stemmer("russian")
    return " ".join(_THREAD.stemmer.stemWords(words))


@dataclass(frozen=True)
class Piece:
    span_id: int
    start: int
    end: int
    text: str
    heading: str


def chunk_span(span: dict, heading: str, embedder):
    text = span["text"]
    # Split at article/chapter boundaries before applying a size window.
    boundaries = [0] + [m.start() for m in HEADING.finditer(text) if m.start() > 0] + [len(text)]
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        match = HEADING.match(text[left:right])
        if match:
            heading = match.group(1).strip()
        start = left
        while start < right:
            end = min(start + 900, right)
            if end < right:
                cut = max(text.rfind("\n", start + 450, end), text.rfind(". ", start + 450, end))
                if cut > start:
                    end = cut + 1
            while embedder.token_count(heading + "\n" + text[start:end]) > embedder.max_tokens:
                if end - start < 16:
                    raise ValueError("heading_or_chunk_exceeds_model_context")
                end = start + (end - start) // 2
            if text[start:end].strip():
                yield Piece(span["id"], start, end, text[start:end], heading)
            if end == right:
                break
            start = max(start + 1, end - min(120, (end - start) // 4))
