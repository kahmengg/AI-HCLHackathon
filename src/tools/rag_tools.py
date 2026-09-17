from src.receiver.retriever import retrieve


def search_documents_tool(
    query: str,
    client_id: str | None = None,
    k: int = 5,
) -> list[dict]:
    """
    Search unstructured documents using query fusion.

    Both the original and rewritten query are searched.
    Their candidate results are merged and reranked.
    """

    # Do not hard-filter by client ID here.
    #
    # Questions about a client may also require global
    # documents such as policies and product factsheets.
    where = None

    return retrieve(
        question=query,
        k=k,
        where=where,

        # Standalone reranking is handled automatically
        # inside query fusion.
        use_reranker=False,

        candidate_k=10,

        # Do not replace the original query.
        use_query_rewrite=False,

        # Search original + rewritten query.
        use_query_fusion=True,

        client_id=client_id,
    )