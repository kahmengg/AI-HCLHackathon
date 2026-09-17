from src.receiver.retriever import retrieve
from src.llm.groq_client import generate


def build_context(
    results: list[dict]
) -> str:
    """
    Convert retrieved Chroma results into an evidence block
    that can be supplied to the LLM.
    """

    context_parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "unknown"
        )

        page = metadata.get(
            "page",
            "n/a"
        )

        client_id = metadata.get(
            "client_id",
            "n/a"
        )

        doc_type = metadata.get(
            "doc_type",
            "unknown"
        )

        text = result.get(
            "text",
            ""
        )

        context_parts.append(
            f"""
SOURCE {index}
source: {source}
page: {page}
client_id: {client_id}
doc_type: {doc_type}

{text}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


def answer_question(
    question: str,
    client_id: str | None = None,
    k: int = 5,
) -> str:
    """
    RAG answer flow:

        question
            ↓
        semantic retrieval
            ↓
        retrieved evidence
            ↓
        grounded prompt
            ↓
        Groq LLM
            ↓
        final answer
    """

    if not question.strip():
        return "Please provide a question."

    # --------------------------------------------------------
    # Optional client filtering
    # --------------------------------------------------------

    where = None

    if client_id:
        where = {
            "client_id": client_id
        }

    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------

    results = retrieve(
        question=question,
        k=k,
        where=where,
    )

    if not results:
        return (
            "I do not have enough evidence in the supplied "
            "documents to answer this question."
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Grounded RAG prompt
    # --------------------------------------------------------

    prompt = f"""
Answer the user's question using ONLY the supplied evidence.

RULES

1. Do not use outside knowledge.

2. Do not invent or assume facts that are not present in the evidence.

3. If the evidence does not support a conclusion, explicitly say that
   there is insufficient evidence.

4. Cite important factual claims using the supplied source information.

5. Prefer direct evidence over indirect evidence.

6. If two sources conflict, explain the conflict instead of choosing one
   without justification.

7. Keep the answer concise and useful for a wealth-management relationship
   manager.

Citation format:

[source: filename, page X]

If there is no page number:

[source: filename]

USER QUESTION

{question}


EVIDENCE

{context}


FINAL ANSWER
""".strip()

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    return generate(
        prompt
    )