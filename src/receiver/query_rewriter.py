from src.llm.groq_client import generate


def rewrite_query(
    question: str,
    client_id: str | None = None,
) -> str:
    """
    Rewrite a conversational user question into a concise
    search query for semantic retrieval.

    Important:
    - This function does NOT answer the question.
    - It must preserve names, IDs, product names and key concepts.
    - It must not invent facts.
    """

    if not question.strip():
        return ""

    client_context = ""

    if client_id:
        client_context = (
            f"\nKnown client ID: {client_id}"
        )

    prompt = f"""
Rewrite the user's question into a concise semantic-search query
for a wealth-management document retrieval system.

Rules:
- Do NOT answer the question.
- Do NOT explain anything.
- Output ONLY the rewritten search query.
- Preserve names, client IDs, product names, dates and important terminology.
- Add useful synonyms or closely related search terms only when they are
  clearly implied by the user's question.
- Do NOT invent facts that are not present in the question.
- Keep the query concise.
{client_context}

User question:
{question}

Rewritten search query:
""".strip()

    try:
        rewritten = generate(prompt)

        rewritten = rewritten.strip()

        if not rewritten:
            return question

        return rewritten

    except Exception:
        # Retrieval should still work even if the rewriting
        # model is temporarily unavailable.
        return question