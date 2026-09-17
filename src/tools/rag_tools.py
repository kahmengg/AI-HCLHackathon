from src.receiver.retriever import retrieve


MAX_CHUNK_CHARS = 1600
MAX_QUERIES = 3
RESULTS_PER_QUERY = 3


def _result_key(item: dict) -> tuple:
    """
    Build a stable key used to remove duplicate chunks returned
    by multiple related search queries.
    """

    metadata = item.get("metadata", {})

    return (
        metadata.get("source", ""),
        metadata.get("page", ""),
        item.get("text", "")[:300],
    )


def _compact_result(
    item: dict,
    matched_query: str,
) -> dict:
    """
    Convert a retrieval result into a compact evidence record
    suitable for sending to the LLM.
    """

    metadata = item.get(
        "metadata",
        {},
    )

    text = item.get(
        "text",
        "",
    )

    if len(text) > MAX_CHUNK_CHARS:
        text = (
            text[:MAX_CHUNK_CHARS]
            + "\n[Chunk truncated]"
        )

    return {
        "matched_query": matched_query,
        "source": metadata.get(
            "source",
            "unknown",
        ),
        "page": metadata.get(
            "page",
            "n/a",
        ),
        "client_id": metadata.get(
            "client_id",
            "n/a",
        ),
        "doc_type": metadata.get(
            "doc_type",
            "unknown",
        ),
        "text": text,
        "rerank_score": item.get(
            "rerank_score",
        ),
    }


def search_documents_tool(
    queries: list[str],
    client_id: str | None = None,
) -> list[dict]:
    """
    Search documents using one or more focused semantic queries.

    The LLM may break a complex or multi-part user question into
    several focused retrieval queries. Each query is searched
    independently using semantic retrieval + CrossEncoder reranking.

    The results are then merged and deduplicated.

    This allows one agent tool call to retrieve evidence for
    multiple parts of a question.
    """

    if not queries:
        return []

    # Prevent excessive retrieval work.
    queries = [
        str(query).strip()
        for query in queries[:MAX_QUERIES]
        if str(query).strip()
    ]

    merged_results = []
    seen = set()

    for query in queries:

        results = retrieve(
            question=query,
            k=RESULTS_PER_QUERY,

            # Do not hard-filter by client because some relevant
            # policy/factsheet documents have client_id = n/a.
            where=None,

            # Evaluated production default.
            use_reranker=True,
            candidate_k=10,

            # These were experimentally tested but are not enabled
            # by default because reranking alone performed best
            # overall in the labelled evaluation.
            use_query_rewrite=False,
            use_query_fusion=False,
            use_doc_type_routing=False,

            client_id=client_id,
        )

        for item in results:

            key = _result_key(
                item
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            merged_results.append(
                _compact_result(
                    item=item,
                    matched_query=query,
                )
            )

    return merged_results