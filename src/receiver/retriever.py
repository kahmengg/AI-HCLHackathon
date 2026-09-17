"""
Semantic retrieval layer.

Supported retrieval modes:

1. Baseline
   Original query -> Chroma

2. Reranking
   Original query -> Chroma candidates -> CrossEncoder reranker

3. Query rewriting
   Rewritten query -> Chroma

4. Query fusion
   Original query
        +
   Rewritten query
        ↓
   merge candidates
        ↓
   remove duplicates
        ↓
   rerank
        ↓
   final evidence
"""

from src.embedding.embedder import embed
from src.vectorstore.chroma_store import query
from src.receiver.reranker import rerank
from src.receiver.query_rewriter import rewrite_query


def _semantic_search(
    search_query: str,
    k: int,
    where: dict | None = None,
) -> list[dict]:
    """
    Perform one semantic Chroma search.
    """

    if not search_query.strip():
        return []

    query_embedding = embed(
        [search_query]
    )[0]

    raw_results = query(
        query_embedding=query_embedding,
        k=k,
        where=where,
    )

    documents = raw_results.get(
        "documents",
        [[]],
    )[0]

    metadatas = raw_results.get(
        "metadatas",
        [[]],
    )[0]

    distances = raw_results.get(
        "distances",
        [[]],
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


def _result_key(
    result: dict,
) -> tuple:
    """
    Build a stable identity for a retrieved chunk.

    Used to remove duplicates when the original query and
    rewritten query retrieve the same evidence.
    """

    metadata = result.get(
        "metadata",
        {},
    )

    return (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("thread_id"),
        metadata.get("message_index"),
        metadata.get("chunk_index"),
        result.get("text", ""),
    )


def _merge_results(
    first_results: list[dict],
    second_results: list[dict],
) -> list[dict]:
    """
    Merge two retrieval result sets without duplicate chunks.

    If the same chunk appears in both searches, preserve its
    best (lowest) Chroma distance.
    """

    merged = {}

    for result in (
        first_results
        + second_results
    ):

        key = _result_key(
            result
        )

        if key not in merged:

            merged[key] = result

        else:

            existing = merged[key]

            if (
                result.get(
                    "distance",
                    float("inf"),
                )
                <
                existing.get(
                    "distance",
                    float("inf"),
                )
            ):
                merged[key] = result

    return list(
        merged.values()
    )


def retrieve(
    question: str,
    k: int = 5,
    where: dict | None = None,
    use_reranker: bool = False,
    candidate_k: int = 10,
    use_query_rewrite: bool = False,
    use_query_fusion: bool = False,
    client_id: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant evidence.

    Parameters
    ----------
    question:
        Original user question.

    k:
        Number of final results.

    where:
        Optional Chroma metadata filter.

    use_reranker:
        Rerank candidate results using the CrossEncoder.

    candidate_k:
        Number of candidates retrieved before reranking.

    use_query_rewrite:
        Replace the original query with an LLM-rewritten query.

        Mainly retained for evaluation/testing.

    use_query_fusion:
        Search BOTH the original and rewritten query,
        merge candidates, then rerank.

        This is the preferred advanced retrieval mode.

    client_id:
        Optional contextual client ID passed to the rewriter.
    """

    if not question.strip():
        return []

    # ======================================================
    # QUERY FUSION
    # ======================================================

    if use_query_fusion:

        rewritten_query = rewrite_query(
            question=question,
            client_id=client_id,
        )

        # Search using the user's exact question.
        original_results = _semantic_search(
            search_query=question,
            k=candidate_k,
            where=where,
        )

        # Search again using the rewritten query.
        rewritten_results = _semantic_search(
            search_query=rewritten_query,
            k=candidate_k,
            where=where,
        )

        # Combine and remove duplicate chunks.
        candidates = _merge_results(
            original_results,
            rewritten_results,
        )

        # Add debugging / explainability information.
        for result in candidates:

            result["original_query"] = (
                question
            )

            result["search_query"] = (
                rewritten_query
            )

            result["retrieval_mode"] = (
                "query_fusion"
            )

        # Reranking is especially useful here because the
        # candidate pool came from two searches.
        reranked_results = rerank(
            question=question,
            results=candidates,
            top_k=k,
        )

        return reranked_results

    # ======================================================
    # SINGLE-QUERY RETRIEVAL
    # ======================================================

    search_query = question

    if use_query_rewrite:

        search_query = rewrite_query(
            question=question,
            client_id=client_id,
        )

    # Retrieve more candidates if reranking is enabled.
    retrieval_k = (
        max(candidate_k, k)
        if use_reranker
        else k
    )

    results = _semantic_search(
        search_query=search_query,
        k=retrieval_k,
        where=where,
    )

    for result in results:

        result["original_query"] = (
            question
        )

        result["search_query"] = (
            search_query
        )

        if use_query_rewrite:
            result["retrieval_mode"] = (
                "rewritten"
            )
        else:
            result["retrieval_mode"] = (
                "baseline"
            )

    # ======================================================
    # OPTIONAL RERANKING
    # ======================================================

    if use_reranker:

        return rerank(
            question=question,
            results=results,
            top_k=k,
        )

    return results[:k]