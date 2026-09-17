import json
from pathlib import Path

from src.receiver.retriever import retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_SET_PATH = (
    PROJECT_ROOT
    / "src"
    / "evaluation"
    / "eval_set.json"
)


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evidence_matches(
    retrieved_result: dict,
    expected: dict,
) -> bool:
    metadata = retrieved_result.get("metadata", {})
    text = retrieved_result.get("text", "").lower()

    if metadata.get("source") != expected.get("source"):
        return False

    optional_fields = [
        "page",
        "thread_id",
        "message_index",
        "chunk_index",
        "client_id",
    ]

    for field in optional_fields:
        if field in expected:
            if metadata.get(field) != expected.get(field):
                return False

    required_phrases = expected.get("must_contain", [])

    for phrase in required_phrases:
        if phrase.lower() not in text:
            return False

    return True


def calculate_recall(
    results: list[dict],
    expected_evidence: list[dict],
    k: int,
) -> float:
    if not expected_evidence:
        return 0.0

    top_k_results = results[:k]

    matched = 0

    for expected in expected_evidence:
        found = any(
            evidence_matches(result, expected)
            for result in top_k_results
        )

        if found:
            matched += 1

    return matched / len(expected_evidence)


def evaluate_query(
    item: dict,
    use_reranker: bool = False,
    use_query_rewrite: bool = False,
    use_query_fusion: bool = False,
    use_doc_type_routing: bool = False,
    max_k: int = 5,
) -> dict:
    question = item["question"]
    client_id = item.get("client_id")
    expected_evidence = item.get("expected_evidence", [])

    results = retrieve(
        question=question,
        k=max_k,
        where=None,
        use_reranker=use_reranker,
        candidate_k=10,
        use_query_rewrite=use_query_rewrite,
        use_query_fusion=use_query_fusion,
        use_doc_type_routing=use_doc_type_routing,
        client_id=client_id,
    )

    return {
        "id": item["id"],
        "question": question,
        "results": results,
        "expected_evidence": expected_evidence,
        "recall_at_1": calculate_recall(
            results,
            expected_evidence,
            1,
        ),
        "recall_at_3": calculate_recall(
            results,
            expected_evidence,
            3,
        ),
        "recall_at_5": calculate_recall(
            results,
            expected_evidence,
            5,
        ),
    }


def print_expected_matches(
    result: dict,
) -> None:
    expected_evidence = result["expected_evidence"]
    retrieved_results = result["results"]

    print()
    print("Expected evidence checks:")

    for index, expected in enumerate(
        expected_evidence,
        start=1,
    ):
        matched_rank = None

        for rank, retrieved in enumerate(
            retrieved_results,
            start=1,
        ):
            if evidence_matches(retrieved, expected):
                matched_rank = rank
                break

        source = expected.get("source", "unknown")
        phrases = expected.get("must_contain", [])

        if matched_rank is not None:
            print(
                f"  Expected {index}: "
                f"HIT at rank {matched_rank} "
                f"| source={source}"
            )
        else:
            print(
                f"  Expected {index}: "
                f"MISS "
                f"| source={source}"
            )

        if phrases:
            print(
                f"      must_contain={phrases}"
            )


def evaluate_mode(
    eval_set: list[dict],
    mode_name: str,
    use_reranker: bool = False,
    use_query_rewrite: bool = False,
    use_query_fusion: bool = False,
    use_doc_type_routing: bool = False,
) -> list[dict]:
    evaluated = []

    print()
    print("=" * 100)
    print(mode_name)
    print("=" * 100)

    for item in eval_set:
        result = evaluate_query(
            item=item,
            use_reranker=use_reranker,
            use_query_rewrite=use_query_rewrite,
            use_query_fusion=use_query_fusion,
            use_doc_type_routing=use_doc_type_routing,
            max_k=5,
        )

        evaluated.append(result)

        print()
        print("-" * 100)

        print(f"ID: {result['id']}")
        print(f"Question: {result['question']}")

        print(
            f"Recall@1: {result['recall_at_1']:.2f}"
        )
        print(
            f"Recall@3: {result['recall_at_3']:.2f}"
        )
        print(
            f"Recall@5: {result['recall_at_5']:.2f}"
        )

        print()
        print("Top retrieved evidence:")

        for index, retrieved in enumerate(
            result["results"],
            start=1,
        ):
            metadata = retrieved.get("metadata", {})

            line = (
                f"{index}. "
                f"{metadata.get('source', 'unknown')} "
                f"| page={metadata.get('page', 'n/a')} "
                f"| client={metadata.get('client_id', 'n/a')} "
                f"| distance={retrieved.get('distance', 0):.4f}"
            )

            if "rerank_score" in retrieved:
                line += (
                    f" | rerank_score="
                    f"{retrieved['rerank_score']:.4f}"
                )

            if "retrieval_mode" in retrieved:
                line += (
                    f" | mode="
                    f"{retrieved['retrieval_mode']}"
                )

            print(line)

        print_expected_matches(result)

    return evaluated


def average_metric(
    evaluated: list[dict],
    key: str,
) -> float:
    if not evaluated:
        return 0.0

    return sum(
        item[key]
        for item in evaluated
    ) / len(evaluated)


def get_summary(
    evaluated: list[dict],
) -> dict:
    return {
        "recall_at_1": average_metric(
            evaluated,
            "recall_at_1",
        ),
        "recall_at_3": average_metric(
            evaluated,
            "recall_at_3",
        ),
        "recall_at_5": average_metric(
            evaluated,
            "recall_at_5",
        ),
    }


def run_evaluation():
    eval_set = load_eval_set()

    if not eval_set:
        print("No evaluation questions found.")
        return

    baseline = evaluate_mode(
        eval_set=eval_set,
        mode_name="BASELINE RETRIEVAL",
    )

    reranked = evaluate_mode(
        eval_set=eval_set,
        mode_name="RERANKED RETRIEVAL",
        use_reranker=True,
    )

    rewritten_reranked = evaluate_mode(
        eval_set=eval_set,
        mode_name="QUERY REWRITE + RERANKED RETRIEVAL",
        use_reranker=True,
        use_query_rewrite=True,
    )

    fusion_reranked = evaluate_mode(
        eval_set=eval_set,
        mode_name="QUERY FUSION + RERANKED RETRIEVAL",
        use_query_fusion=True,
    )

    routed_reranked = evaluate_mode(
        eval_set=eval_set,
        mode_name="DOC-TYPE ROUTING + RERANKED RETRIEVAL",
        use_doc_type_routing=True,
    )

    baseline_summary = get_summary(baseline)
    reranked_summary = get_summary(reranked)
    rewritten_summary = get_summary(rewritten_reranked)
    fusion_summary = get_summary(fusion_reranked)
    routed_summary = get_summary(routed_reranked)

    print()
    print("=" * 125)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 125)

    print(
        f"{'Metric':<15}"
        f"{'Baseline':<18}"
        f"{'Reranked':<18}"
        f"{'Rewrite+Rerank':<20}"
        f"{'Fusion+Rerank':<20}"
        f"{'Routing+Rerank':<20}"
    )

    print("-" * 111)

    metrics = [
        ("Recall@1", "recall_at_1"),
        ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"),
    ]

    for label, key in metrics:
        print(
            f"{label:<15}"
            f"{baseline_summary[key]:<18.2%}"
            f"{reranked_summary[key]:<18.2%}"
            f"{rewritten_summary[key]:<20.2%}"
            f"{fusion_summary[key]:<20.2%}"
            f"{routed_summary[key]:<20.2%}"
        )


if __name__ == "__main__":
    run_evaluation()