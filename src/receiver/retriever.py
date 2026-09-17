"""
Retrieval layer for the RAG system.

Flow:

    user question
        ↓
    embed question
        ↓
    query ChromaDB
        ↓
    retrieve top-k relevant chunks
        ↓
    return clean results

This file does NOT call the LLM yet.
Its only responsibility is retrieving relevant evidence.
"""

from src.embedding.embedder import embed
from src.vectorstore.chroma_store import query


def retrieve(
    question: str,
    k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Retrieve the most semantically relevant document chunks.

    Parameters
    ----------
    question:
        Natural-language user question.

    k:
        Number of results to retrieve.

    where:
        Optional Chroma metadata filter.

        Examples:

            {"client_id": "CL002"}

            {"doc_type": "fund_factsheet"}

    Returns
    -------
    A list of dictionaries such as:

        [
            {
                "text": "...",
                "metadata": {...},
                "distance": 0.12
            }
        ]
    """

    if not question.strip():
        return []

    # --------------------------------------------------------
    # 1. Embed the user's question
    # --------------------------------------------------------

    query_embedding = embed(
        [question]
    )[0]

    # --------------------------------------------------------
    # 2. Search Chroma
    # --------------------------------------------------------

    raw_results = query(
        query_embedding=query_embedding,
        k=k,
        where=where,
    )

    # --------------------------------------------------------
    # 3. Convert Chroma's result format into something easier
    #    for the rest of our application to use.
    # --------------------------------------------------------

    documents = raw_results.get(
        "documents",
        [[]]
    )[0]

    metadatas = raw_results.get(
        "metadatas",
        [[]]
    )[0]

    distances = raw_results.get(
        "distances",
        [[]]
    )[0]

    results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        results.append(
            {
                "text": document,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )

    return results