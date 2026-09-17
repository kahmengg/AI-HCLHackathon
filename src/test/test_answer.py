from src.agent.answer import answer_question


if __name__ == "__main__":

    question = (
        "How much LRS remittance headroom does Arjun Mehta (CL015) have left for the current financial year as of his July 2026 discussion with his RM, and can his planned USD 60,000 top-up proceed as-is?"
    )

    answer = answer_question(
        question=question,
        k=5,
    )

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)