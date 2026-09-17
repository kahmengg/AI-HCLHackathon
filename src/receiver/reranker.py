from sentence_transformers import CrossEncoder


_reranker = None

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    """
    Load the reranking model once and reuse it.

    A CrossEncoder looks at the question and passage together,
    making it more precise than embedding similarity alone.
    """

    global _reranker

    if _reranker is None:
        print(
            f"Loading reranker: {RERANKER_MODEL}"
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

    return _reranker


def rerank(
    question: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved Chroma candidates using a CrossEncoder.

    Parameters
    ----------
    question:
        The original user question.

    results:
        Candidate chunks retrieved from Chroma.

    top_k:
        Number of reranked results to return.

    Returns
    -------
    Results sorted from highest reranker relevance score
    to lowest.
    """

    if not results:
        return []

    model = _get_reranker()

    pairs = [
        (
            question,
            result.get("text", "")
        )
        for result in results
    ]

    scores = model.predict(
        pairs
    )

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):
        updated_result = {
            **result,
            "rerank_score": float(score),
        }

        reranked_results.append(
            updated_result
        )

    reranked_results.sort(
        key=lambda item: item[
            "rerank_score"
        ],
        reverse=True,
    )

    return reranked_results[:top_k]