"""
Thin wrapper around a persistent ChromaDB collection. Kept deliberately
small so the architecture diagram / README can point at one file and say
"this is the vector store boundary" - no framework abstraction to unpack.
"""
import chromadb
from src.config import CHROMA_DIR, COLLECTION_NAME

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection():
    return _get_client().get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(ids: list[str], documents: list[str],
               embeddings: list[list[float]], metadatas: list[dict]):
    get_collection().add(ids=ids, documents=documents,
                          embeddings=embeddings, metadatas=metadatas)


def query(query_embedding: list[float], k: int = 5, where: dict | None = None):
    """Returns Chroma's raw result dict: documents, metadatas, distances (lower = closer)."""
    return get_collection().query(
        query_embeddings=[query_embedding], n_results=k, where=where
    )


def count() -> int:
    return get_collection().count()
