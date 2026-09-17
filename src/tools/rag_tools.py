from src.receiver.retriever import retrieve


def search_documents_tool(
    query: str,
    client_id: str | None = None,
    k: int = 5,
) -> list[dict]:
    """
    Search unstructured documents using the selected
    production retrieval pipeline:

    semantic retrieval -> CrossEncoder reranking.
    """

    return retrieve(
        question=query,
        k=k,
        where=None,

        # Selected default based on evaluation
        use_reranker=True,
        candidate_k=10,

        # Tested but not enabled by default
        use_query_rewrite=False,
        use_query_fusion=False,
        use_doc_type_routing=False,

        client_id=client_id,
    )