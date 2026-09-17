"""
Persistent ChromaDB vector store.

Stores:
- text chunks
- embedding vectors
- metadata

Provides semantic similarity retrieval over unstructured documents.
"""

import chromadb

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
)


_client = None


def _get_client():
    """
    Create the persistent Chroma client once.
    """

    global _client

    if _client is None:

        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

    return _client


def get_collection():
    """
    Get or create the document collection.

    Cosine distance is used for embedding similarity.
    """

    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )


def add_chunks(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
):
    """
    Insert/update chunks in ChromaDB.
    """

    if not ids:
        return

    get_collection().upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(
    query_embedding: list[float],
    k: int = 5,
    where: dict | None = None,
):
    """
    Retrieve the k most semantically similar chunks.

    Lower cosine distance means greater similarity.
    """

    kwargs = {
        "query_embeddings": [
            query_embedding
        ],
        "n_results": k,
    }

    if where is not None:
        kwargs["where"] = where

    return get_collection().query(
        **kwargs
    )


def count() -> int:
    """
    Number of chunks currently stored.
    """

    return get_collection().count()