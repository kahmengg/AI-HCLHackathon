"""
Extract text from PDFs while preserving page numbers and useful metadata.

Each PDF page becomes one record.

The loader also attempts to detect client IDs such as CL002 from page text
so later retrieval can filter by client_id.
"""

import re
from pathlib import Path

from pypdf import PdfReader


CLIENT_ID_PATTERN = re.compile(
    r"\bCL\d{3}\b",
    re.IGNORECASE,
)


def infer_doc_type(filename: str) -> str:
    name = filename.lower()

    if name.startswith("fund_factsheet"):
        return "fund_factsheet"

    if name.startswith("policy"):
        return "policy"

    if "complaint" in name:
        return "complaint_letter"

    if "call_notes" in name:
        return "rm_call_notes"

    if "risk_acknowledgement" in name:
        return "risk_acknowledgement_form"

    return "other"


def extract_client_ids(text: str) -> list[str]:
    """
    Extract client IDs such as CL002 from page text.
    """

    matches = CLIENT_ID_PATTERN.findall(text)

    # Normalise to uppercase and remove duplicates.
    return sorted(
        set(
            match.upper()
            for match in matches
        )
    )


def load_pdf(path: Path) -> list[dict]:
    """
    Return one record per PDF page.

    Each record contains:
    - source
    - doc_type
    - client_id
    - page
    - text
    """

    reader = PdfReader(str(path))

    records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = (
            page.extract_text()
            or ""
        ).strip()

        if not text:
            continue

        client_ids = extract_client_ids(text)

        # If exactly one client appears on the page,
        # assign that client directly.
        if len(client_ids) == 1:
            client_id = client_ids[0]

        else:
            # Multiple clients or none.
            client_id = "n/a"

        records.append(
            {
                "source": path.name,
                "doc_type": infer_doc_type(
                    path.name
                ),
                "client_id": client_id,
                "page": page_number,
                "text": text,
            }
        )

    if not records:
        print(
            f"[WARN] No extractable text in "
            f"{path.name}. PDF may be scanned."
        )

    return records


def load_all_pdfs(
    data_dir: Path
) -> list[dict]:

    records = []

    for pdf_path in sorted(
        data_dir.glob("*.pdf")
    ):

        pdf_records = load_pdf(
            pdf_path
        )

        records.extend(
            pdf_records
        )

    return records