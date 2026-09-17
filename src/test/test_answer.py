from src.agent.answer import answer_question


if __name__ == "__main__":

    question = (
        "What complaint did Robert Chua make "
        "about the APEX autocallable note?"
    )

    answer = answer_question(
        question=question,
        client_id="CL002",
        k=5,
    )

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)