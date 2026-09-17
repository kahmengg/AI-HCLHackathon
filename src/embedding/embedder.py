"""
Embedding wrapper. Two backends behind one `embed()` function so swapping
doesn't touch any other file:

  - "local"  -> sentence-transformers, runs offline, no API key, slightly
                lower quality. Good default for a hackathon with no
                guaranteed API budget.
  - "openai" -> text-embedding-3-small, needs OPENAI_API_KEY, higher quality,
                especially on domain terms (fund names, ISINs).

Switch via EMBEDDING_BACKEND in src/config.py.
"""
from src.config import EMBEDDING_BACKEND, LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL

_local_model = None
_openai_client = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_model


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()  # reads OPENAI_API_KEY from env
    return _openai_client


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Returns one vector per input string, same order."""
    if EMBEDDING_BACKEND == "openai":
        client = _get_openai_client()
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    # default: local
    model = _get_local_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
