from src.receiver.retriever import retrieve


if __name__ == "__main__":

    question = (
        "What complaint did Robert Chua make "
        "about the APEX autocallable note?"
    )

    print()
    print("=" * 80)
    print("WITHOUT RERANKING")
    print("=" * 80)

    normal_results = retrieve(
        question=question,
        k=5,
        where={
            "client_id": "CL002"
        },
        use_reranker=False,
    )

    for index, result in enumerate(
        normal_results,
        start=1,
    ):
        metadata = result["metadata"]

        print(
            f"{index}. "
            f"{metadata.get('source')} "
            f"| page={metadata.get('page', 'n/a')} "
            f"| distance={result['distance']:.4f}"
        )

    print()
    print("=" * 80)
    print("WITH RERANKING")
    print("=" * 80)

    reranked_results = retrieve(
        question=question,
        k=5,
        where={
            "client_id": "CL002"
        },
        use_reranker=True,
        candidate_k=10,
    )

    for index, result in enumerate(
        reranked_results,
        start=1,
    ):
        metadata = result["metadata"]

        print(
            f"{index}. "
            f"{metadata.get('source')} "
            f"| page={metadata.get('page', 'n/a')} "
            f"| distance={result['distance']:.4f} "
            f"| rerank_score="
            f"{result['rerank_score']:.4f}"
        )