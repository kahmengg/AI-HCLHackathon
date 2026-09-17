from src.receiver.retriever import retrieve


TESTS = [
    {
        "name": "Robert policy",
        "question": (
            "Is Robert Chua's (CL002) holding of the "
            "APEX Global Multi-Asset Autocallable Note Series 7 "
            "suitable given his risk profile? "
            "What documentation is missing?"
        ),
        "look_for": [
            "Risk Score of 7 or above",
        ],
    },
    {
        "name": "Park policy",
        "question": (
            "Does Park Ji-hoon's (CL011) 35% portfolio allocation "
            "to the APEX Autocallable Note breach firm policy?"
        ),
        "look_for": [
            "20%",
            "Retail Investor",
        ],
    },
    {
        "name": "Fund SRI",
        "question": (
            "What is the Summary Risk Indicator (SRI) rating "
            "of the APAC Stable Income Money Market Fund, "
            "and what is the minimum initial investment "
            "for a retail investor?"
        ),
        "look_for": [
            "SUMMARY RISK INDICATOR",
            "1 / 7",
        ],
    },
]


def contains_all(
    text: str,
    phrases: list[str],
) -> bool:

    lowered = text.lower()

    return all(
        phrase.lower() in lowered
        for phrase in phrases
    )


if __name__ == "__main__":

    for test in TESTS:

        print()
        print("=" * 100)
        print(test["name"])
        print("=" * 100)

        print(
            f"Question: {test['question']}"
        )

        # -------------------------------------------------
        # Vector search only — NO reranker
        # -------------------------------------------------

        results = retrieve(
            question=test["question"],
            k=20,
            where=None,
            use_reranker=False,
            candidate_k=20,
            use_query_rewrite=False,
            use_query_fusion=False,
        )

        found_rank = None

        for rank, result in enumerate(
            results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            text = result.get(
                "text",
                "",
            )

            is_expected = contains_all(
                text,
                test["look_for"],
            )

            if is_expected:
                found_rank = rank

            marker = (
                "  <<< EXPECTED"
                if is_expected
                else ""
            )

            print(
                f"{rank:>2}. "
                f"{metadata.get('source', 'unknown')} "
                f"| page={metadata.get('page', 'n/a')} "
                f"| distance={result.get('distance', 0):.4f}"
                f"{marker}"
            )

        print()

        if found_rank is None:

            print(
                "EXPECTED CHUNK NOT FOUND "
                "IN TOP 20"
            )

        else:

            print(
                f"EXPECTED CHUNK FOUND "
                f"AT VECTOR RANK #{found_rank}"
            )