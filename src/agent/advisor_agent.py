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
MAX_DOCUMENT_SEARCHES = 1
MAX_TOOL_RESULT_CHARS = 7000
MAX_RESPONSE_TOKENS = 700


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
                        "description": "Client ID such as CL002",
                    }
                },
                "required": ["client_id"],
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
                "required": ["client_id"],
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
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search unstructured evidence such as complaints, "
                "client correspondence, policy documents and factsheets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                    "client_id": {
                        "type": ["string", "null"],
                    },
                    "k": {
                        "type": "integer",
                        "default": 5,
                    },
                },
                "required": ["query"],
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
            query=arguments["query"],
            client_id=arguments.get("client_id"),
            k=arguments.get("k", 5),
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

    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            content[:MAX_TOOL_RESULT_CHARS]
            + "\n\n[Tool result truncated for context size.]"
        )

    return content


def build_final_answer(
    question: str,
    messages: list[dict],
) -> str:
    """
    Produce a final grounded answer after the tool-call limit
    has been reached.

    A clean conversation is used so the model does not try
    to call another tool.
    """

    client = get_client()

    evidence_parts = []

    for item in messages:
        if item.get("role") == "tool":
            content = item.get("content", "")

            if content:
                evidence_parts.append(content)

    evidence_text = "\n\n".join(
        evidence_parts
    )

    if len(evidence_text) > 12000:
        evidence_text = evidence_text[:12000]

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded wealth-advisor assistant. "
                "You cannot use tools in this response. "
                "Answer only from the evidence supplied below. "
                "Do not invent facts, dates, amounts, policies, "
                "or client information. "
                "If the requested information is not present, "
                "clearly state that the available records do not "
                "provide enough information to answer. "
                "Do not request or attempt another tool call. "
                "When source metadata is available, cite the actual "
                "filename and page, for example "
                "[rm_call_notes_log.pdf, p.1]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{question}\n\n"
                f"Available evidence:\n{evidence_text}\n\n"
                "Provide the final answer using only this evidence."
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

                "Do not recommend buy, sell, unwind, redeem, rebalance, "
                "or other investment actions unless retrieved evidence "
                "explicitly contains that recommendation or the user "
                "explicitly asks for advice. "

                "Do not invent facts, next steps, dates, amounts, "
                "client details, policy requirements, or missing information. "

                "If the available tools do not contain the requested "
                "information, explicitly state that the available records "
                "do not provide enough evidence to answer. "

                "Do not repeatedly search for information already shown "
                "to be unavailable. "

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
            response.choices[0].message
        )

        messages.append(
            message.model_dump(
                exclude_none=True
            )
        )

        # ----------------------------------------------------
        # No more tools requested -> final answer
        # ----------------------------------------------------

        if not message.tool_calls:
            return message.content or ""

        # ----------------------------------------------------
        # Execute requested tools
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = json.loads(
                tool_call.function.arguments
            )

            signature = make_tool_signature(
                tool_name,
                arguments,
            )

            # ------------------------------------------------
            # Prevent identical repeated calls
            # ------------------------------------------------

            if signature in called_tools:

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {
                                "status": (
                                    "duplicate_tool_call_skipped"
                                ),
                                "message": (
                                    "This exact tool call was already "
                                    "executed. Use the evidence already "
                                    "returned. If it does not contain the "
                                    "requested information, state that the "
                                    "available records are insufficient."
                                ),
                            }
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # Limit document searches to one per question
            # ------------------------------------------------

            if tool_name == "search_documents":

                if (
                    document_search_count
                    >= MAX_DOCUMENT_SEARCHES
                ):

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "status": (
                                        "document_search_limit_reached"
                                    ),
                                    "message": (
                                        "Document search has already "
                                        "been performed for this question. "
                                        "Use the evidence already returned. "
                                        "If the requested information is "
                                        "not present, state that the "
                                        "available records are insufficient."
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