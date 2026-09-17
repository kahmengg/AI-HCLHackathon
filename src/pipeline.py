from pathlib import Path

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.json_loader import load_correspondence
from src.ingestion.chunker import chunk_record

from src.embedding.embedder import embed

from src.vectorstore.chroma_store import (
    add_chunks,
    count,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# LOAD UNSTRUCTURED DATA
# ============================================================

def load_unstructured_data() -> list[dict]:
    """
    Load all unstructured records that belong in the RAG pipeline.

    Includes:
    - PDFs
    - client correspondence JSON

    Structured portfolio/transaction data is NOT loaded here.
    """

    records = []

    # --------------------------------------------------------
    # PDFs
    # --------------------------------------------------------

    pdf_records = load_all_pdfs(DATA_DIR)

    records.extend(pdf_records)

    # --------------------------------------------------------
    # Client correspondence
    # --------------------------------------------------------

    correspondence_path = (
        DATA_DIR / "client_correspondence.json"
    )

    correspondence_records = load_correspondence(
        correspondence_path
    )

    records.extend(correspondence_records)

    return records


# ============================================================
# CHUNKING
# ============================================================

def chunk_all_records(
    records: list[dict]
) -> list[dict]:
    """
    Chunk every loaded document/message while preserving metadata.
    """

    chunks = []

    for record in records:
        record_chunks = chunk_record(record)

        chunks.extend(record_chunks)

    return chunks


# ============================================================
# BUILD UNIQUE CHROMA IDs
# ============================================================

def build_chunk_id(
    chunk: dict,
    row_index: int,
) -> str:
    """
    Build a deterministic unique ID for each Chroma record.

    PDFs use:
        source + page + chunk index

    Correspondence additionally uses:
        thread_id + message_index

    row_index is included as a final safety measure so two different
    records in the same ingestion batch can never have the same ID.
    """

    source = str(
        chunk.get("source", "unknown")
    )

    client_id = str(
        chunk.get("client_id", "na")
    )

    thread_id = str(
        chunk.get("thread_id", "na")
    )

    message_index = str(
        chunk.get("message_index", "na")
    )

    page = str(
        chunk.get("page", "na")
    )

    chunk_index = str(
        chunk.get("chunk_index", "na")
    )

    return (
        f"{source}"
        f"-client-{client_id}"
        f"-thread-{thread_id}"
        f"-msg-{message_index}"
        f"-page-{page}"
        f"-chunk-{chunk_index}"
        f"-row-{row_index}"
    )


# ============================================================
# BUILD CHROMA METADATA
# ============================================================

def build_metadata(
    chunk: dict
) -> dict:
    """
    Convert chunk metadata into a Chroma-safe dictionary.

    The actual text is stored separately as the Chroma document.
    """

    metadata = {}

    for key, value in chunk.items():

        if key == "text":
            continue

        if value is None:
            continue

        # Chroma metadata should contain simple scalar values.
        if isinstance(
            value,
            (str, int, float, bool)
        ):
            metadata[key] = value

        else:
            # Convert unexpected metadata types to strings
            # rather than failing ingestion.
            metadata[key] = str(value)

    return metadata


# ============================================================
# EMBEDDING + CHROMA STORAGE
# ============================================================

def embed_and_store(
    chunks: list[dict]
) -> None:
    """
    Embed all chunks and store them in ChromaDB.
    """

    if not chunks:
        print("No chunks found. Nothing to embed.")
        return

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Embedding {len(documents)} chunks..."
    )

    # --------------------------------------------------------
    # Generate vectors
    # --------------------------------------------------------

    embeddings = embed(documents)

    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Embedding count does not match document count."
        )

    # --------------------------------------------------------
    # Generate unique IDs
    # --------------------------------------------------------

    ids = [
        build_chunk_id(
            chunk,
            row_index=index,
        )
        for index, chunk in enumerate(chunks)
    ]

    # Defensive duplicate check before Chroma
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate Chroma IDs were generated."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadatas = [
        build_metadata(chunk)
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    add_chunks(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Stored/upserted {len(chunks)} chunks."
    )

    print(
        f"ChromaDB currently contains "
        f"{count()} chunks."
    )


# ============================================================
# OPTIONAL DEBUG OUTPUT
# ============================================================

def print_sample_chunks(
    chunks: list[dict],
    limit: int = 3,
) -> None:
    """
    Print a few chunks so ingestion can be inspected manually.
    """

    print("\nSample chunks:")

    for chunk in chunks[:limit]:
        print("\n----------------")
        print(chunk)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> None:

    print("\n==============================")
    print("1. Loading unstructured data")
    print("==============================")

    records = load_unstructured_data()

    print(
        f"Loaded {len(records)} records"
    )


    print("\n==============================")
    print("2. Chunking")
    print("==============================")

    chunks = chunk_all_records(records)

    print(
        f"Created {len(chunks)} chunks"
    )

    print_sample_chunks(
        chunks,
        limit=3,
    )


    print("\n==============================")
    print("3. Embedding + ChromaDB")
    print("==============================")

    embed_and_store(chunks)


    print("\n==============================")
    print("Pipeline complete")
    print("==============================")


if __name__ == "__main__":
    run_pipeline()