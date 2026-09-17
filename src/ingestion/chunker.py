"""
Structure-aware chunking for unstructured documents.

Strategy:

1. Detect real document section headings where possible.
2. Keep sections together when they are small.
3. Split large sections into overlapping word windows.
4. Preserve metadata such as source, page, client_id, etc.

This avoids blindly cutting documents at fixed positions while also
preventing identifiers such as CPL-2026-014 from being mistaken for
section headings.
"""

import re

from src.config import (
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)


# Common headings expected in wealth-management documents.
KNOWN_HEADINGS = {
    "OBJECTIVE",
    "INVESTMENT OBJECTIVE",
    "RISK FACTORS",
    "RISKS",
    "KEY RISKS",
    "PERFORMANCE",
    "PAST PERFORMANCE",
    "SUITABILITY",
    "PRODUCT FEATURES",
    "PRODUCT DETAILS",
    "IMPORTANT INFORMATION",
    "FEES",
    "CHARGES",
    "TERMS AND CONDITIONS",
    "CLIENT ACKNOWLEDGEMENT",
    "RISK ACKNOWLEDGEMENT",
    "COMPLAINT DETAILS",
    "BACKGROUND",
    "RECOMMENDATION",
    "CONCLUSION",
}


def _is_heading(line: str) -> bool:
    """
    Determine whether a line is likely to be a real section heading.

    Avoid treating identifiers such as:
        CPL-2026-014
        TXN-1001

    as headings.
    """

    line = line.strip()

    if not line:
        return False

    # Explicit known headings are always accepted.
    if line.upper() in KNOWN_HEADINGS:
        return True

    # Keep generic heading detection conservative.
    if len(line) > 70:
        return False

    # IDs containing lots of numbers are not section headings.
    digit_count = sum(char.isdigit() for char in line)

    if digit_count >= 2:
        return False

    # Must contain alphabetic text.
    if not any(char.isalpha() for char in line):
        return False

    # Generic heading must be uppercase.
    if line != line.upper():
        return False

    # Only allow reasonable heading characters.
    if not re.fullmatch(
        r"[A-Z][A-Z /&,\-()]+",
        line,
    ):
        return False

    return True


def split_into_sections(text: str) -> list[str]:
    """
    Split text around detected section headings.

    If no headings are found, return the entire text as one section.
    """

    lines = text.splitlines()

    sections = []
    current = []

    found_heading = False

    for line in lines:

        if _is_heading(line):

            found_heading = True

            if current:
                section = "\n".join(current).strip()

                if section:
                    sections.append(section)

            current = [line]

        else:
            current.append(line)

    if current:
        section = "\n".join(current).strip()

        if section:
            sections.append(section)

    if not found_heading:
        return [text.strip()] if text.strip() else []

    return sections


def _sliding_window(
    words: list[str],
    size: int,
    overlap: int,
):
    """
    Yield overlapping windows of words.

    Example:
        size = 300
        overlap = 50

        chunk 1 = words 1-300
        chunk 2 = words 251-550
    """

    step = max(size - overlap, 1)

    index = 0

    while index < len(words):

        yield words[index:index + size]

        if index + size >= len(words):
            break

        index += step


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split document text into manageable chunks.
    """

    chunks = []

    sections = split_into_sections(text)

    for section in sections:

        words = section.split()

        if not words:
            continue

        # Keep short sections together.
        if len(words) <= chunk_size:

            chunks.append(section)

        else:

            # Split large sections with overlap.
            for window in _sliding_window(
                words,
                chunk_size,
                overlap,
            ):

                chunks.append(
                    " ".join(window)
                )

    return chunks


def chunk_record(
    record: dict,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """
    Chunk one loader record while preserving its metadata.

    Input example:

        {
            "source": "fund_factsheet.pdf",
            "page": 4,
            "doc_type": "fund_factsheet",
            "text": "..."
        }

    Output:

        [
            {
                "source": "fund_factsheet.pdf",
                "page": 4,
                "doc_type": "fund_factsheet",
                "chunk_index": 0,
                "text": "..."
            }
        ]
    """

    text = record.get(
        "text",
        ""
    )

    text_chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    results = []

    for index, text_chunk in enumerate(
        text_chunks
    ):

        results.append(
            {
                **record,
                "chunk_index": index,
                "text": text_chunk,
            }
        )

    return results