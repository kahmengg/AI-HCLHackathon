from src.agent.advisor_agent import run_agent


if __name__ == "__main__":

    question = (
        "Does Robert Chua's APEX autocallable note "
        "raise any suitability concerns?"
    )

    answer = run_agent(question)

    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)