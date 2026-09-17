import json

from src.llm.groq_client import get_client, DEFAULT_MODEL

from src.tools.structured_tools import (
    client_profile_tool,
    portfolio_tool,
    transactions_tool,
    suitability_mismatches_tool,
)

from src.tools.rag_tools import (
    search_documents_tool,
)


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


def run_agent(
    question: str,
) -> str:

    client = get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a wealth-advisor assistant operating only on information "
                "returned by the provided tools. "
                "Use tools whenever factual client, portfolio, transaction, product, "
                "policy, complaint, or correspondence information is required. "
                "Do not use outside knowledge or general financial knowledge as evidence. "
                "Every factual claim must be supported by tool output. "
                "Clearly separate retrieved facts from interpretation. "
                "Do not recommend buy, sell, unwind, redeem, rebalance, or other investment "
                "actions unless the retrieved evidence explicitly contains that recommendation "
                "or the user explicitly asks for advice. "
                "Do not invent next steps. "
                "If evidence is insufficient, say what is missing. "
                "Cite the supporting source or tool for important claims."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    while True:

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )

        message = response.choices[0].message

        messages.append(
            message.model_dump(
                exclude_none=True
            )
        )

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = json.loads(
                tool_call.function.arguments
            )

            result = execute_tool(
                tool_name,
                arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )