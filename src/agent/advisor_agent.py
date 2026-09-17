import json

from src.llm.groq_client import (
    get_client,
    DEFAULT_MODEL,
)

from src.tools.structured_tools import (
    client_profile_tool,
    portfolio_tool,
    transactions_tool,
)

from src.tools.rag_tools import (
    search_documents_tool,
)


# ============================================================
# Agent settings
# ============================================================

MAX_TOOL_ROUNDS = 3

# Only one document-search TOOL CALL is allowed.
# That single call may contain up to 3 focused queries.
MAX_DOCUMENT_SEARCHES = 1

MAX_TOOL_RESULT_CHARS = 12000
MAX_RESPONSE_TOKENS = 900


# ============================================================
# Tool definitions
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_client_profile",
            "description": (
                "Get structured client information such as "
                "risk profile, risk score, AUM and investment objective."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": (
                            "Client ID such as CL002."
                        ),
                    }
                },
                "required": [
                    "client_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": (
                "Get the client's current portfolio holdings "
                "and allocations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                    }
                },
                "required": [
                    "client_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": (
                "Get transaction history for a client."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                    }
                },
                "required": [
                    "client_id"
                ],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search unstructured evidence such as fund factsheets, "
                "policy documents, complaints, RM call notes and client "
                "correspondence. "
                "For a simple question, provide one focused search query. "
                "For a multi-part or complex question, split the information "
                "need into 2 or at most 3 focused semantic search queries "
                "and provide them together in the queries array. "
                "Each query should preserve important product names, client "
                "names, client IDs, figures and concepts from the user's "
                "question. Do not answer the question inside the search query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "minItems": 1,
                        "maxItems": 3,
                        "description": (
                            "One to three focused semantic retrieval queries. "
                            "Use multiple queries when the user asks for "
                            "multiple distinct facts."
                        ),
                    },
                    "client_id": {
                        "type": [
                            "string",
                            "null",
                        ],
                        "description": (
                            "Relevant client ID if known, such as CL002. "
                            "May be null for product or policy questions."
                        ),
                    },
                },
                "required": [
                    "queries"
                ],
            },
        },
    },
]


# ============================================================
# Tool execution
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict,
):

    if tool_name == "get_client_profile":

        return client_profile_tool(
            arguments["client_id"]
        )

    if tool_name == "get_portfolio":

        return portfolio_tool(
            arguments["client_id"]
        )

    if tool_name == "get_transactions":

        return transactions_tool(
            arguments["client_id"]
        )

    if tool_name == "search_documents":

        return search_documents_tool(
            queries=arguments["queries"],
            client_id=arguments.get(
                "client_id"
            ),
        )

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# ============================================================
# Helpers
# ============================================================

def make_tool_signature(
    tool_name: str,
    arguments: dict,
) -> str:

    return (
        f"{tool_name}:"
        f"{json.dumps(arguments, sort_keys=True)}"
    )


def serialize_tool_result(
    result,
) -> str:

    content = json.dumps(
        result,
        default=str,
        ensure_ascii=False,
    )

    if (
        len(content)
        > MAX_TOOL_RESULT_CHARS
    ):

        content = (
            content[
                :MAX_TOOL_RESULT_CHARS
            ]
            + "\n\n"
            + "[Tool result truncated for context size.]"
        )

    return content


def build_final_answer(
    question: str,
    messages: list[dict],
) -> str:
    """
    Produce a final grounded answer after the normal
    tool-call loop has finished.

    A clean conversation is used so the model cannot
    attempt another tool call.
    """

    client = get_client()

    evidence_parts = []

    for item in messages:

        if item.get("role") == "tool":

            content = item.get(
                "content",
                "",
            )

            if content:

                evidence_parts.append(
                    content
                )

    evidence_text = "\n\n".join(
        evidence_parts
    )

    if len(evidence_text) > 16000:

        evidence_text = (
            evidence_text[:16000]
        )

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded wealth-advisor assistant. "

                "You cannot use tools in this response. "

                "Answer only from the supplied evidence. "

                "Do not invent facts, dates, amounts, policies, "
                "client information, conclusions, or missing details. "

                "When evidence contains ambiguity, exceptions, "
                "qualifications, or pending decisions, preserve that "
                "ambiguity rather than resolving it yourself. "

                "If the requested information is not contained in "
                "the supplied evidence, state that the available "
                "records do not provide enough information to answer. "

                "When source metadata is available, cite the actual "
                "filename and page, for example "
                "[rm_call_notes_log.pdf, p.1]."
            ),
        },

        {
            "role": "user",
            "content": (
                f"Original question:\n"
                f"{question}\n\n"

                f"Available evidence:\n"
                f"{evidence_text}\n\n"

                "Answer every part of the original question using "
                "only the evidence above."
            ),
        },
    ]

    final_response = (
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=final_messages,
            temperature=0.1,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
    )

    return (
        final_response
        .choices[0]
        .message
        .content
        or ""
    )


# ============================================================
# Agent
# ============================================================

def run_agent(
    question: str,
) -> str:

    client = get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a wealth-advisor assistant operating only "
                "on information returned by the provided tools. "

                "Use tools whenever factual client, portfolio, "
                "transaction, product, policy, complaint, or "
                "correspondence information is required. "

                "Do not use outside knowledge or general financial "
                "knowledge as factual evidence. "

                "Every factual claim must be supported by tool output. "
                "Clearly separate retrieved facts from interpretation. "

                "When using search_documents, identify the actual "
                "information needs in the user's question. "
                "For a simple question, use one focused semantic query. "
                "For a multi-part question, create separate focused queries "
                "for the distinct facts being requested and send them "
                "together in ONE search_documents call. "
                "Use no more than three queries. "
                "Preserve important client names, IDs, product names and "
                "specific concepts in those search queries. "

                "For example, if a question asks for both a fund's SRI "
                "and minimum investment, search separately for the fund's "
                "SRI and the fund's minimum investment within the same "
                "search_documents tool call. "

                "When evidence contains an exception, qualification, pending "
                "compliance decision, unresolved interpretation, or ambiguity, "
                "do not resolve that ambiguity yourself. State clearly what is "
                "known and what remains undetermined. "

                "Do not convert a suitability concern, suitability flag, "
                "potential mismatch, complaint, or pending compliance review "
                "into a definitive finding that a product is unsuitable unless "
                "the retrieved evidence explicitly records that conclusion. "

                "Never use definitive conclusions such as 'is suitable', "
                "'is not suitable', 'unsuitable', 'mis-sold', 'policy breach', "
                "or 'non-compliant' based only on your own interpretation. "
                "Only use such a definitive conclusion when retrieved evidence "
                "explicitly records that conclusion as a completed determination. "
                "Otherwise describe the evidence as raising a suitability concern, "
                "potential mismatch, or matter requiring review, using terminology "
                "supported by the retrieved records. "

                "When a client states that they did not understand a product, "
                "did not recall a risk being explained, or was confused about "
                "product mechanics, report that fact accurately. "
                "Do not independently conclude that disclosure was inadequate "
                "or that the client was mis-sold the product unless retrieved "
                "evidence explicitly makes that finding. "

                "Do not recommend buy, sell, unwind, redeem, rebalance, "
                "or other investment actions unless retrieved evidence "
                "explicitly contains that recommendation or the user "
                "explicitly asks for advice. "

                "Do not infer that a threshold has been breached merely because "
                "an amount or percentage appears large. "
                "Only state a threshold breach when retrieved evidence establishes "
                "the applicable threshold, establishes that the threshold applies "
                "to that client, and shows that the threshold has actually "
                "been exceeded. "

                "Do not describe an allocation as excessive, above policy limits, "
                "or non-compliant unless retrieved evidence supports that statement. "

                "Do not treat missing documentation as proof that an assessment, "
                "review, acknowledgement, disclosure, or process was never completed "
                "unless retrieved evidence explicitly states that it was not completed. "

                "If evidence says that a document cannot be located or is not "
                "on file, report exactly that level of certainty. "

                "Do not invent facts, next steps, dates, amounts, "
                "client details, policy requirements, or missing information. "

                "If the available tools do not contain the requested "
                "information, explicitly state that the available records "
                "do not provide enough evidence to answer. "

                "Do not repeatedly search for information already shown "
                "to be unavailable. "

                "When answering a multi-part question, answer each requested "
                "part separately. Do not declare one part unavailable merely "
                "because evidence for another part was easier to find. "

                "For document evidence, cite the actual source metadata "
                "whenever available. Prefer citations such as "
                "[rm_call_notes_log.pdf, p.1] or "
                "[client_complaint_letters.pdf, p.2]. "

                "For structured data, cite the relevant structured tool "
                "or source, for example [get_client_profile, CL002]. "

                "Do not cite search_documents as the source when an actual "
                "filename and page are available."
            ),
        },

        {
            "role": "user",
            "content": question,
        },
    ]

    called_tools = set()

    document_search_count = 0

    # ========================================================
    # Tool-call loop
    # ========================================================

    for tool_round in range(
        MAX_TOOL_ROUNDS
    ):

        response = (
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
        )

        message = (
            response
            .choices[0]
            .message
        )

        messages.append(
            message.model_dump(
                exclude_none=True
            )
        )

        # ----------------------------------------------------
        # No tools requested -> final answer
        # ----------------------------------------------------

        if not message.tool_calls:

            return (
                message.content
                or ""
            )

        # ----------------------------------------------------
        # Execute requested tools
        # ----------------------------------------------------

        for tool_call in (
            message.tool_calls
        ):

            tool_name = (
                tool_call
                .function
                .name
            )

            arguments = json.loads(
                tool_call
                .function
                .arguments
            )

            # ------------------------------------------------
            # Normalise document search arguments
            # ------------------------------------------------

            if (
                tool_name
                == "search_documents"
            ):

                queries = (
                    arguments.get(
                        "queries",
                        [],
                    )
                )

                # Defensive fallback in case the model ever returns
                # a single string instead of an array.
                if isinstance(
                    queries,
                    str,
                ):

                    queries = [
                        queries
                    ]

                queries = [
                    str(query).strip()
                    for query
                    in queries[:3]
                    if str(query).strip()
                ]

                if not queries:

                    queries = [
                        question
                    ]

                arguments[
                    "queries"
                ] = queries

            signature = (
                make_tool_signature(
                    tool_name,
                    arguments,
                )
            )

            # ------------------------------------------------
            # Prevent identical repeated calls
            # ------------------------------------------------

            if signature in called_tools:

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": json.dumps(
                            {
                                "status": (
                                    "duplicate_tool_call_skipped"
                                ),
                                "message": (
                                    "This exact tool call was already "
                                    "executed. Use the evidence already "
                                    "returned. If the requested information "
                                    "is not present, state that the available "
                                    "records are insufficient."
                                ),
                            }
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # Limit document-search tool calls
            # ------------------------------------------------

            if (
                tool_name
                == "search_documents"
            ):

                if (
                    document_search_count
                    >= MAX_DOCUMENT_SEARCHES
                ):

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": (
                                tool_call.id
                            ),
                            "content": json.dumps(
                                {
                                    "status": (
                                        "document_search_limit_reached"
                                    ),
                                    "message": (
                                        "Document retrieval has already "
                                        "been performed for this question. "
                                        "Use the evidence already returned. "
                                        "If a requested fact is absent, "
                                        "state that the available records "
                                        "are insufficient."
                                    ),
                                }
                            ),
                        }
                    )

                    continue

                document_search_count += 1

            # ------------------------------------------------
            # Execute tool
            # ------------------------------------------------

            called_tools.add(
                signature
            )

            result = execute_tool(
                tool_name,
                arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "content": (
                        serialize_tool_result(
                            result
                        )
                    ),
                }
            )

    # ========================================================
    # Maximum tool rounds reached
    # ========================================================

    return build_final_answer(
        question=question,
        messages=messages,
    )