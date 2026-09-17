"""
Function-calling tools exposed to the LLM. This satisfies the brief's
"function tools" mandatory requirement, and gives you two clearly separated
capabilities instead of one fuzzy "search everything" call:

  - retrieve_documents          -> semantic search (policies, fact sheets,
                                    call notes, correspondence)
  - query_portfolio              -> exact holdings lookup for one client
  - query_transactions           -> exact transaction ledger lookup
  - query_concentration_breaches -> exact rule check across all clients

Wire these into your LLM call's `tools` parameter (Anthropic/OpenAI function-
calling format) - the docstrings here double as a starting point for your
tool-use JSON schema descriptions.
"""
from src.config import DEFAULT_TOP_K
from src.embedding.embedder import embed
from src.vectorstore.chroma_store import query as vector_query
from src.structured.structured_data import (
    get_portfolio,
    get_transactions,
    get_concentration_breaches,
)


def retrieve_documents(query_text: str, k: int = DEFAULT_TOP_K,
                        doc_type: str | None = None) -> list[dict]:
    """Semantic search over ingested PDFs/correspondence.

    Args:
        query_text: natural-language question or topic.
        k: number of chunks to return.
        doc_type: optional filter, e.g. "fund_factsheet", "policy",
                  "rm_call_notes", "complaint_letter", "correspondence".
    Returns:
        list of {text, metadata, distance} - lower distance = more relevant.
        Use `distance` for abstention logic (see src/agent/answer.py).
    """
    query_embedding = embed([query_text])[0]
    where = {"doc_type": doc_type} if doc_type else None
    results = vector_query(query_embedding, k=k, where=where)

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def query_portfolio(client_id: str) -> list[dict]:
    """Exact portfolio holdings for one client_id."""
    return get_portfolio(client_id).to_dict(orient="records")


def query_transactions(client_id: str, start_date: str | None = None,
                        end_date: str | None = None) -> list[dict]:
    """Exact transaction ledger rows for one client_id, optional date range (YYYY-MM-DD)."""
    return get_transactions(client_id, start_date, end_date).to_dict(orient="records")


def query_concentration_breaches(threshold_pct: float = 20.0) -> list[dict]:
    """Clients whose single holding exceeds threshold_pct of their total portfolio."""
    return get_concentration_breaches(threshold_pct).to_dict(orient="records")
