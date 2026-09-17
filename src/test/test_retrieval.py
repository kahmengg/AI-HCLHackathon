from src.receiver.retriever import retrieve


def print_results(
    question: str,
    results: list[dict]
):
    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    print("\n")
    print("=" * 80)
    print("RETRIEVED RESULTS")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print(
            f"\nRESULT #{index}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "unknown"
            )
        )

        print(
            "Document type:",
            metadata.get(
                "doc_type",
                "unknown"
            )
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "n/a"
            )
        )

        print(
            "Client:",
            metadata.get(
                "client_id",
                "n/a"
            )
        )

        print("\nTEXT:")

        print(
            result["text"]
        )

        print(
            "\n" + "-" * 80
        )


if __name__ == "__main__":

    question = (
        "What complaint did Robert Chua "
        "make about the APEX autocallable note?"
    )

    results = retrieve(
        question=question,
        k=5,    
        where={
        "client_id": "CL002"
    },
    )

    print_results(
        question,
        results,
        
    )