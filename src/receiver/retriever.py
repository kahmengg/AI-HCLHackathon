"""
Semantic retrieval layer with optional reranking,
query rewriting, query fusion, and document-type routing.
"""

from src.embedding.embedder import embed
from src.vectorstore.chroma_store import query
from src.receiver.reranker import rerank
from src.receiver.query_rewriter import rewrite_query


POLICY_KEYWORDS = {
    "policy",
    "breach",
    "suitable",
    "suitability",
    "risk profile",
    "complex product",
    "compliance",
    "risk acknowledgement",
}


def _semantic_search(
    search_query: str,
    k: int,
    where: dict | None = None,
) -> list[dict]:

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
    *result_sets: list[dict],
) -> list[dict]:

    merged = {}

    for result_set in result_sets:

        for result in result_set:

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


def _looks_like_policy_question(
    question: str,
) -> bool:

    lowered = question.lower()

    return any(
        keyword in lowered
        for keyword in POLICY_KEYWORDS
    )


def retrieve(
    question: str,
    k: int = 5,
    where: dict | None = None,
    use_reranker: bool = False,
    candidate_k: int = 10,
    use_query_rewrite: bool = False,
    use_query_fusion: bool = False,
    use_doc_type_routing: bool = False,
    client_id: str | None = None,
) -> list[dict]:

    if not question.strip():
        return []

    # ======================================================
    # DOCUMENT-TYPE ROUTING
    # ======================================================

    if use_doc_type_routing:

        global_results = _semantic_search(
            search_query=question,
            k=candidate_k,
            where=where,
        )

        routed_results = []

        if _looks_like_policy_question(
            question
        ):
            routed_results = _semantic_search(
                search_query=question,
                k=candidate_k,
                where={
                    "doc_type": "policy"
                },
            )

        candidates = _merge_results(
            global_results,
            routed_results,
        )

        for result in candidates:

            result["original_query"] = (
                question
            )

            result["search_query"] = (
                question
            )

            result["retrieval_mode"] = (
                "doc_type_routing"
            )

        return rerank(
            question=question,
            results=candidates,
            top_k=k,
        )

    # ======================================================
    # QUERY FUSION
    # ======================================================

    if use_query_fusion:

        rewritten_query = rewrite_query(
            question=question,
            client_id=client_id,
        )

        original_results = _semantic_search(
            search_query=question,
            k=candidate_k,
            where=where,
        )

        rewritten_results = _semantic_search(
            search_query=rewritten_query,
            k=candidate_k,
            where=where,
        )

        candidates = _merge_results(
            original_results,
            rewritten_results,
        )

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

        return rerank(
            question=question,
            results=candidates,
            top_k=k,
        )

    # ======================================================
    # SINGLE-QUERY RETRIEVAL
    # ======================================================

    search_query = question

    if use_query_rewrite:

        search_query = rewrite_query(
            question=question,
            client_id=client_id,
        )

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

        result["retrieval_mode"] = (
            "rewritten"
            if use_query_rewrite
            else "baseline"
        )

    if use_reranker:

        return rerank(
            question=question,
            results=results,
            top_k=k,
        )

    return results[:k]