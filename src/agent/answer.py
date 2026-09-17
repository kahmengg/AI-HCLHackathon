"""
Agent loop using Claude's tool-use API. The model decides which tool(s) to
call (retrieve_documents / query_portfolio / query_transactions /
query_concentration_breaches), we execute them locally, feed results back,
and repeat until it produces a final answer. This is what turns "plain RAG"
into the "agentic RAG" bonus item - the model can chain multiple retrievals
(e.g. fund fact sheet, then a specific client's portfolio) before answering.

Grounding + abstention is enforced in the SYSTEM_PROMPT, not bolted on after
the fact - the model is told exactly what "insufficient evidence" means and
what phrase to use, so abstention is a text pattern you can grade against in
your eval harness (see eval/run_eval.py).
"""
import json
from anthropic import Anthropic
from src.agent.tools import (
    retrieve_documents, query_portfolio, query_transactions, query_concentration_breaches,
)

client = Anthropic()  # reads ANTHROPIC_API_KEY from env
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a Wealth Advisor Assistant for relationship managers.

RULES (do not break these):
1. Answer ONLY using information returned by your tools in this conversation.
   Never use outside knowledge about funds, clients, or policy.
2. Every factual claim must cite its source, e.g. "(source: fund_factsheet_safe.pdf)"
   or "(source: clients_portfolio.csv)".
3. If the tools do not return enough evidence to answer confidently, you MUST
   respond with exactly this phrase at the start of your answer:
   "INSUFFICIENT EVIDENCE:" followed by a one-sentence explanation of what's
   missing. Do not guess or fill gaps with general knowledge.
4. For questions about a specific client, always call query_portfolio and/or
   query_transactions - do not rely on document retrieval alone for numbers.
5. For questions about fund/policy content, call retrieve_documents.
"""

TOOL_DEFINITIONS = [
    {
        "name": "retrieve_documents",
        "description": "Semantic search over policies, fund fact sheets, RM call notes, complaint letters, and client correspondence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "doc_type": {
                    "type": "string",
                    "enum": ["fund_factsheet", "policy", "rm_call_notes",
                              "complaint_letter", "correspondence", "risk_acknowledgement_form"],
                    "description": "Optional filter to one document type.",
                },
            },
            "required": ["query_text"],
        },
    },
    {
        "name": "query_portfolio",
        "description": "Exact portfolio holdings for one client_id.",
        "input_schema": {
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
        },
    },
    {
        "name": "query_transactions",
        "description": "Exact transaction ledger rows for one client_id, optional date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["client_id"],
        },
    },
    {
        "name": "query_concentration_breaches",
        "description": "Clients whose single holding exceeds a concentration threshold percent.",
        "input_schema": {
            "type": "object",
            "properties": {"threshold_pct": {"type": "number"}},
            "required": [],
        },
    },
]

_DISPATCH = {
    "retrieve_documents": retrieve_documents,
    "query_portfolio": query_portfolio,
    "query_transactions": query_transactions,
    "query_concentration_breaches": query_concentration_breaches,
}


def _run_tool(name: str, tool_input: dict):
    fn = _DISPATCH[name]
    return fn(**tool_input)


def answer_question(question: str, max_turns: int = 4) -> dict:
    """Runs the agent loop and returns {answer, tool_calls} for logging/eval."""
    messages = [{"role": "user", "content": question}]
    tool_call_log = []

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return {"answer": final_text, "tool_calls": tool_call_log}

        # Execute every tool_use block, feed results back as tool_result blocks
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _run_tool(block.name, block.input)
            tool_call_log.append({"tool": block.name, "input": block.input})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "INSUFFICIENT EVIDENCE: reached max tool-call turns without a resolved answer.",
            "tool_calls": tool_call_log}
