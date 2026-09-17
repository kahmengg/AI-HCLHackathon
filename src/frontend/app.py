import streamlit as st

from src.agent.answer import answer_question
from src.receiver.retriever import retrieve


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Wealth Advisor Assistant",
    page_icon="💼",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Wealth Advisor Assistant")

st.caption(
    "RAG-powered assistant for client portfolios, "
    "complaints, correspondence and investment documents."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Search Settings")

    client_id = st.text_input(
        "Client ID",
        placeholder="e.g. CL002",
        help=(
            "Optional. Restricts document retrieval "
            "to a specific client."
        ),
    )

    top_k = st.slider(
        "Retrieved documents",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.subheader("System")

    st.write("Structured data: SQLite")
    st.write("Vector database: ChromaDB")
    st.write("Embeddings: MiniLM")
    st.write("LLM: Groq")

    st.divider()

    st.caption(
        "Answers are generated only from retrieved evidence."
    )


# ---------------------------------------------------------
# Example questions
# ---------------------------------------------------------

st.subheader("Example Questions")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button(
        "Robert Chua complaint",
        use_container_width=True,
    ):
        st.session_state["question"] = (
            "What complaint did Robert Chua make "
            "about the APEX autocallable note?"
        )
        st.session_state["example_client"] = "CL002"

with example_col2:
    if st.button(
        "Risk acknowledgement",
        use_container_width=True,
    ):
        st.session_state["question"] = (
            "Was there a signed risk acknowledgement "
            "for Robert Chua's APEX note?"
        )
        st.session_state["example_client"] = "CL002"

with example_col3:
    if st.button(
        "Suitability concerns",
        use_container_width=True,
    ):
        st.session_state["question"] = (
            "What evidence suggests there may be "
            "a suitability issue for Robert Chua?"
        )
        st.session_state["example_client"] = "CL002"


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

default_question = st.session_state.get(
    "question",
    "",
)

question = st.text_area(
    "Ask a question",
    value=default_question,
    height=100,
    placeholder=(
        "Example: What complaint did Robert Chua "
        "make about the APEX autocallable note?"
    ),
)


# If an example button supplied a client ID, use it unless
# the user manually entered another one in the sidebar.
effective_client_id = (
    client_id.strip()
    or st.session_state.get(
        "example_client",
        "",
    )
)


# ---------------------------------------------------------
# Ask button
# ---------------------------------------------------------

ask = st.button(
    "Ask Assistant",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Main RAG execution
# ---------------------------------------------------------

if ask:

    if not question.strip():

        st.warning(
            "Enter a question first."
        )

    else:

        where = None

        if effective_client_id:
            where = {
                "client_id": effective_client_id
            }

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        with st.status(
            "Searching documents...",
            expanded=False,
        ) as status:

            results = retrieve(
                question=question,
                k=top_k,
                where=where,
            )

            status.update(
                label=(
                    f"Retrieved {len(results)} "
                    f"relevant chunks"
                ),
                state="complete",
            )

        # -------------------------------------------------
        # Answer generation
        # -------------------------------------------------

        with st.spinner(
            "Generating grounded answer..."
        ):

            try:

                answer = answer_question(
                    question=question,
                    client_id=(
                        effective_client_id
                        if effective_client_id
                        else None
                    ),
                    k=top_k,
                )

            except Exception as exc:

                st.error(
                    f"Unable to generate answer: {exc}"
                )

                st.stop()

        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        st.divider()

        st.subheader("Answer")

        st.markdown(answer)

        # -------------------------------------------------
        # Retrieved evidence
        # -------------------------------------------------

        st.divider()

        st.subheader("Retrieved Evidence")

        if not results:

            st.warning(
                "No supporting evidence was retrieved."
            )

        else:

            for index, result in enumerate(
                results,
                start=1,
            ):

                metadata = result.get(
                    "metadata",
                    {},
                )

                source = metadata.get(
                    "source",
                    "Unknown source",
                )

                page = metadata.get(
                    "page",
                    "n/a",
                )

                doc_type = metadata.get(
                    "doc_type",
                    "unknown",
                )

                retrieved_client = metadata.get(
                    "client_id",
                    "n/a",
                )

                distance = result.get(
                    "distance",
                    0,
                )

                title = (
                    f"Evidence {index} — {source}"
                )

                if page != "n/a":
                    title += f" — Page {page}"

                with st.expander(title):

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Similarity distance",
                            f"{distance:.4f}",
                        )

                    with col2:
                        st.metric(
                            "Document type",
                            doc_type,
                        )

                    with col3:
                        st.metric(
                            "Client",
                            retrieved_client,
                        )

                    st.markdown(
                        "**Retrieved text**"
                    )

                    st.write(
                        result.get(
                            "text",
                            "",
                        )
                    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Prototype — responses should be reviewed by a human advisor "
    "before being used for client decisions."
)