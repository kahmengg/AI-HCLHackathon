"""
Embedding wrapper.

Converts text into numerical vectors for semantic search.

Supported backends:

- "local":
    Uses sentence-transformers locally.
    No API key required.

- "openai":
    Uses OpenAI embeddings.
    Requires OPENAI_API_KEY.

The rest of the project only calls embed(texts), so the embedding
backend can be changed without modifying the ingestion or retrieval code.
"""

from src.config import (
    EMBEDDING_BACKEND,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
)


_local_model = None
_openai_client = None


def _get_local_model():
    """
    Load the local SentenceTransformer model once.

    The model is cached so it is not reloaded every time embed() is called.
    """

    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(
            LOCAL_EMBEDDING_MODEL
        )

    return _local_model


def _get_openai_client():
    """
    Create the OpenAI client once.

    OpenAI automatically reads OPENAI_API_KEY from the environment.
    """

    global _openai_client

    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()

    return _openai_client


def embed(
    texts: list[str]
) -> list[list[float]]:
    """
    Convert a list of strings into embedding vectors.

    Example:

        texts = [
            "Capital loss may occur.",
            "Client has a conservative risk profile."
        ]

        vectors = embed(texts)

    Returns one vector per input text in the same order.
    """

    if not texts:
        return []

    if EMBEDDING_BACKEND == "openai":

        client = _get_openai_client()

        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts,
        )

        # Preserve API response order explicitly
        ordered = sorted(
            response.data,
            key=lambda item: item.index,
        )

        return [
            item.embedding
            for item in ordered
        ]

    elif EMBEDDING_BACKEND == "local":

        model = _get_local_model()

        vectors = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    else:

        raise ValueError(
            f"Unknown embedding backend: {EMBEDDING_BACKEND}. "
            f"Expected 'local' or 'openai'."
        )