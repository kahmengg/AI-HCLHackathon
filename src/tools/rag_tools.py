from src.receiver.retriever import retrieve


def search_documents_tool(
    query: str,
    client_id: str | None = None,
    k: int = 5,
) -> list[dict]:

    where = None

    if client_id:
        where = {
            "client_id": client_id
        }

    return retrieve(
        question=query,
        k=k,
        where=where,
    )