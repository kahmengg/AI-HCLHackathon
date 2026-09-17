"""
Extract raw text from every PDF in the data folder.

We tag each document with a `doc_type` inferred from its filename - this is
what lets the retriever later filter to e.g. "only fund fact sheets" or
"only policy docs" before/alongside semantic search, instead of relying on
similarity alone.
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


def load_pdf(path: Path) -> dict:
    """Return one record per PDF: {source, doc_type, text}."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return {
        "source": path.name,
        "doc_type": infer_doc_type(path.name),
        "client_id": "n/a",
        "text": "\n".join(pages),
    }


def load_all_pdfs(data_dir: Path) -> list[dict]:
    records = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        record = load_pdf(pdf_path)
        if record["text"].strip():
            records.append(record)
        else:
            # Empty extraction usually means a scanned/image-only PDF - flag
            # it loudly rather than silently ingesting nothing.
            print(f"[WARN] No extractable text in {pdf_path.name} - may need OCR.")
    return records
