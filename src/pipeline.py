"""
Main ingestion entrypoint.

Track 1 (unstructured -> vector store): all PDFs + client_correspondence.json
Track 2 (structured -> DataFrames):      clients_portfolio.csv, transactions.csv

Run from the project root:
    python -m src.pipeline
"""
import uuid

from src.config import DATA_DIR
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.json_loader import load_correspondence
from src.ingestion.chunker import chunk_text
from src.embedding.embedder import embed
from src.vectorstore.chroma_store import add_chunks, count
from src.structured.structured_data import load_structured_data


def ingest_text_track() -> int:
    records = load_all_pdfs(DATA_DIR)
    records.extend(load_correspondence(DATA_DIR / "client_correspondence.json"))

    total_chunks = 0
    for record in records:
        chunks = chunk_text(record["text"])
        if not chunks:
            continue

        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embed(chunks)
        metadatas = [
            {
                "source": record["source"],
                "doc_type": record["doc_type"],
                "client_id": record.get("client_id", "n/a"),
            }
            for _ in chunks
        ]
        add_chunks(ids, chunks, embeddings, metadatas)
        total_chunks += len(chunks)
        print(f"  ingested {len(chunks):>3} chunks  <-  {record['source']}")

    return total_chunks


def ingest_structured_track() -> None:
    portfolio_df, transactions_df = load_structured_data(DATA_DIR)
    print(f"  loaded {len(portfolio_df)} portfolio rows, {len(transactions_df)} transaction rows")


def main():
    print("== Track 1: unstructured docs -> chunk -> embed -> vector store ==")
    total = ingest_text_track()
    print(f"Total chunks in vector store: {count()} (added {total} this run)\n")

    print("== Track 2: structured data -> DataFrames ==")
    ingest_structured_track()

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
