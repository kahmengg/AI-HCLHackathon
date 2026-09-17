from src.receiver.query_rewriter import rewrite_query


if __name__ == "__main__":

    questions = [
        "What happened with Robert?",
        "Was there anything wrong with his APEX thing?",
        "Did he understand what he bought?",
        "What complaint did Robert Chua make about the APEX autocallable note?",
    ]

    for question in questions:

        rewritten = rewrite_query(
            question=question,
            client_id="CL002",
        )

        print()
        print("=" * 80)

        print(
            f"Original:  {question}"
        )

        print(
            f"Rewritten: {rewritten}"
        )