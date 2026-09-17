"""
Extract text from PDFs.

Each PDF page becomes its own document record so page numbers can
be preserved for citations.
"""

from pathlib import Path

from pypdf import PdfReader


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


def load_pdf(path: Path) -> list[dict]:

    reader = PdfReader(str(path))

    records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        records.append(
            {
                "source": path.name,
                "doc_type": infer_doc_type(
                    path.name
                ),
                "client_id": "n/a",
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