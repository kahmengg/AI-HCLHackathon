import sys
from pathlib import Path

import streamlit as st


# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.agent.advisor_agent import run_agent
from src.receiver.retriever import retrieve


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Wealth Advisor Assistant",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# Header
# ============================================================

st.title("Wealth Advisor Assistant")

st.caption(
    "Agentic RAG prototype using structured portfolio data, "
    "document retrieval, reranking, and grounded LLM responses."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Query Settings")

    client_id = st.text_input(
        "Client ID (optional)",
        placeholder="e.g. CL002",
    )

    top_k = st.slider(
        "Retrieved document evidence",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.divider()

    st.subheader("System Components")

    st.markdown(
        """
        - SQLite structured data
        - Chroma vector database
        - MiniLM embeddings
        - CrossEncoder reranking
        - Groq LLM
        - Agentic function tools
        """
    )

    st.divider()

    st.subheader("Example Questions")

    example_questions = [
        (
            "Does Robert Chua's APEX autocallable "
            "note raise any suitability concerns?"
        ),
        (
            "How much LRS remittance headroom does "
            "Arjun Mehta have left?"
        ),
        (
            "What happened with James Sullivan's "
            "request to de-risk his portfolio?"
        ),
        (
            "Does Park Ji-hoon's 35% allocation to "
            "the APEX Autocallable Note breach firm policy?"
        ),
        (
            "What is the SRI rating and minimum investment "
            "for the APAC Stable Income Money Market Fund?"
        ),
    ]

    for index, example in enumerate(
        example_questions,
        start=1,
    ):
        st.caption(
            f"{index}. {example}"
        )


# ============================================================
# User question
# ============================================================

question = st.text_area(
    "Ask a wealth-management question",
    height=120,
    placeholder=(
        "Example: Does Robert Chua's APEX "
        "autocallable note raise any suitability concerns?"
    ),
)


ask_button = st.button(
    "Ask Advisor",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Run agent
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Enter a question first."
        )

    else:

        # ----------------------------------------------------
        # Add optional client context
        # ----------------------------------------------------

        agent_question = question.strip()

        if client_id.strip():

            agent_question += (
                f"\n\nRelevant client ID: "
                f"{client_id.strip()}"
            )

        # ----------------------------------------------------
        # Agent response
        # ----------------------------------------------------

        with st.spinner(
            "Analysing client data and documents..."
        ):

            try:

                answer = run_agent(
                    agent_question
                )

            except Exception as exc:

                st.error(
                    "The advisor agent encountered an error."
                )

                st.exception(exc)
                st.stop()

        # ====================================================
        # Main answer
        # ====================================================

        st.subheader("Advisor Response")

        st.markdown(answer)

        # ====================================================
        # Supporting document retrieval
        # ====================================================

        st.divider()

        st.subheader(
            "Retrieved Document Evidence"
        )

        st.caption(
            "These are the highest-ranked document chunks "
            "returned by semantic retrieval + CrossEncoder "
            "reranking."
        )

        try:

            evidence = retrieve(
                question=question,
                k=top_k,
                where=None,

                # Our evaluated default retrieval configuration
                use_reranker=True,
                candidate_k=10,

                use_query_rewrite=False,
                use_query_fusion=False,
                use_doc_type_routing=False,

                client_id=(
                    client_id.strip()
                    if client_id.strip()
                    else None
                ),
            )

        except Exception as exc:

            st.warning(
                "The answer was generated, but retrieved "
                "evidence could not be displayed."
            )

            st.exception(exc)

            evidence = []

        # ====================================================
        # Evidence display
        # ====================================================

        if not evidence:

            st.info(
                "No document evidence was retrieved."
            )

        else:

            for rank, item in enumerate(
                evidence,
                start=1,
            ):

                metadata = item.get(
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

                evidence_client = metadata.get(
                    "client_id",
                    "n/a",
                )

                rerank_score = item.get(
                    "rerank_score",
                )

                title = (
                    f"#{rank} — {source}"
                )

                if page != "n/a":
                    title += (
                        f" — Page {page}"
                    )

                with st.expander(
                    title,
                    expanded=(
                        rank == 1
                    ),
                ):

                    col1, col2, col3 = st.columns(
                        3
                    )

                    with col1:
                        st.metric(
                            "Rank",
                            rank,
                        )

                    with col2:
                        st.metric(
                            "Client",
                            evidence_client,
                        )

                    with col3:

                        if rerank_score is not None:

                            st.metric(
                                "Rerank Score",
                                f"{rerank_score:.3f}",
                            )

                        else:

                            st.metric(
                                "Rerank Score",
                                "n/a",
                            )

                    st.markdown(
                        "**Retrieved text**"
                    )

                    st.write(
                        item.get(
                            "text",
                            "",
                        )
                    )

        # ====================================================
        # Explainability
        # ====================================================

        st.divider()

        with st.expander(
            "How this answer was produced"
        ):

            st.markdown(
                """
                **1. Agent reasoning**
                
                The advisor agent determines which available
                tools are required for the question.

                **2. Structured data**
                
                Exact client, portfolio, and transaction
                information is retrieved from SQLite when
                required.

                **3. Document retrieval**
                
                Relevant document chunks are retrieved from
                Chroma using semantic vector similarity.

                **4. Reranking**
                
                A CrossEncoder compares the question directly
                against candidate chunks and reorders them by
                relevance.

                **5. Grounded generation**
                
                The LLM is instructed to answer using only
                evidence returned by the tools and to avoid
                inventing unsupported facts.
                """
            )