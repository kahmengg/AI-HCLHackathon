from src.agent.advisor_agent import run_agent


QUESTIONS = [
    (
        "What was Robert Chua's exact annual salary in 2025?"
    ),
    (
        "What is the current market price of the APEX "
        "Autocallable Note?"
    ),
    (
        "What did Park Ji-hoon discuss with his RM "
        "on 15 September 2026?"
    ),
]


if __name__ == "__main__":

    for question in QUESTIONS:

        print()
        print("=" * 90)
        print("QUESTION")
        print("=" * 90)
        print(question)

        print()
        print("=" * 90)
        print("ANSWER")
        print("=" * 90)

        answer = run_agent(question)

        print(answer)