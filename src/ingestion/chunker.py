"""
Structure-aware chunking.

Fund fact sheets and policy docs have clear section headers (OBJECTIVE, RISK
FACTORS, PERFORMANCE, etc). Splitting on those first - instead of a blind
fixed-size window - means a chunk about "risk profile" doesn't get spliced
together with unrelated "past performance" text, which keeps citations
accurate. Sections longer than CHUNK_SIZE_WORDS still get a sliding window
applied within the section so no single chunk is too large to embed well.
"""
import re
from src.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

# Matches short, mostly-uppercase standalone lines - a decent proxy for
# section headers in fact sheets / policy PDFs without needing layout info.
HEADER_PATTERN = re.compile(r"^[ \t]*[A-Z][A-Z0-9 /&,\-]{3,60}[ \t]*$", re.MULTILINE)


def split_into_sections(text: str) -> list[str]:
    headers = list(HEADER_PATTERN.finditer(text))
    if not headers:
        return [text]

    sections = []
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _sliding_window(words: list[str], size: int, overlap: int):
    step = max(size - overlap, 1)
    i = 0
    while i < len(words):
        yield words[i:i + size]
        if i + size >= len(words):
            break
        i += step


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    chunks = []
    for section in split_into_sections(text):
        words = section.split()
        if not words:
            continue
        if len(words) <= chunk_size:
            chunks.append(section)
        else:
            for window in _sliding_window(words, chunk_size, overlap):
                chunks.append(" ".join(window))
    return chunks
