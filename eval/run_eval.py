"""
Eval harness - satisfies two deliverables at once:
  - "at least one retrieval metric such as Recall@K"
  - "retrieval and answer evaluation results"

Expects a golden QA file with entries shaped like:
  {
    "question": "...",
    "expected_sources": ["fund_factsheet_safe.pdf", "clients_portfolio.csv"],
    "expect_abstention": false
  }

The README mentions `golden_qa_dataset.json`; your data folder has
`golden_dataset_for_RAG_evaluation.xlsx` instead - `load_golden_set()` below
handles either, so point GOLDEN_SET_PATH at whichever you actually have.

Run from project root:
    python -m eval.run_eval
"""
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DEFAULT_TOP_K
from src.agent.tools import retrieve_documents

GOLDEN_SET_PATH = DATA_DIR / "golden_dataset_for_RAG_evaluation.xlsx"  # or golden_qa_dataset.json


def load_golden_set(path: Path) -> list[dict]:
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        # NOTE: adjust these column names once you've opened the real file.
        records = []
        for _, row in df.iterrows():
            expected = str(row.get("expected_sources", ""))
            records.append({
                "question": row.get("question", ""),
                "expected_sources": [s.strip() for s in expected.split(",") if s.strip()],
                "expect_abstention": bool(row.get("expect_abstention", False)),
            })
        return records
    raise ValueError(f"Unsupported golden set format: {path.suffix}")


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Fraction of expected sources that appear anywhere in the retrieved set."""
    if not expected_sources:
        return 1.0
    hit = sum(1 for src in expected_sources if any(src in r for r in retrieved_sources))
    return hit / len(expected_sources)


def evaluate_retrieval(golden_set: list[dict], k: int = DEFAULT_TOP_K) -> dict:
    per_question = []
    for item in golden_set:
        results = retrieve_documents(item["question"], k=k)
        retrieved_sources = [r["metadata"]["source"] for r in results]
        score = recall_at_k(retrieved_sources, item["expected_sources"])
        per_question.append({
            "question": item["question"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": retrieved_sources,
            "recall_at_k": score,
        })

    avg_recall = sum(r["recall_at_k"] for r in per_question) / len(per_question) if per_question else 0.0
    return {"avg_recall_at_k": avg_recall, "k": k, "per_question": per_question}


def main():
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    results = evaluate_retrieval(golden_set)

    print(f"Recall@{results['k']}: {results['avg_recall_at_k']:.2f}  "
          f"({len(results['per_question'])} questions)\n")
    for r in results["per_question"]:
        print(f"- {r['question'][:70]}")
        print(f"  expected: {r['expected_sources']}")
        print(f"  retrieved: {r['retrieved_sources']}")
        print(f"  recall@k: {r['recall_at_k']:.2f}\n")

    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
