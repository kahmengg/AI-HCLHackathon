# Wealth Advisor Assistant

RAG copilot for relationship managers: retrieves fund fact sheets, policy
docs, call notes, and client records to answer client-specific questions
with citations, and abstains when evidence is insufficient.

## Architecture

Two ingestion tracks, kept deliberately separate:

```
Unstructured (PDFs, correspondence.json)      Structured (portfolio/txn CSV)
        |                                              |
   extract text                                   pandas DataFrame
        |                                              |
  structure-aware chunk                          (never chunked/embedded -
        |                                         exact lookups only)
   embed (local or OpenAI)                             |
        |                                              |
   ChromaDB (persistent)                                |
        \_______________________  ________________ ____/
                                \/
                    Claude agent (tool-use loop)
                    - retrieve_documents
                    - query_portfolio
                    - query_transactions
                    - query_concentration_breaches
                                |
                    grounded answer + citations
                    or "INSUFFICIENT EVIDENCE: ..."
```

**Why two tracks:** the golden question set needs both concept lookup
("what does the suitability policy say about complex products") and exact
numeric/date reasoning ("clients over the 20% concentration guideline").
Embedding structured data into prose loses exact filtering; keeping the
LLM's tools split by kind fixes that without extra complexity.

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # fill in ANTHROPIC_API_KEY
```

Drop the provided dataset files into `data/` (already matches the folder
you have: PDFs, `clients_portfolio.csv/json`, `transactions.csv`,
`client_correspondence.json`, `golden_dataset_for_RAG_evaluation.xlsx`).

## Before your first real run

1. `python -c "from pathlib import Path; from src.structured.structured_data import inspect_columns; inspect_columns(Path('data'))"`
   -> confirms the real CSV column names. Adjust `CLIENT_ID_COL` /
   `HOLDING_VALUE_COL` / `TXN_DATE_COL` in `src/structured/structured_data.py`
   if they differ.
2. Open `data/client_correspondence.json` once and check it matches the
   `messages`/`body` shape assumed in `src/ingestion/json_loader.py` -
   adjust the `.get(...)` keys there if not.

## Run ingestion

```bash
python -m src.pipeline
```

## Ask a question

```python
from src.agent.answer import answer_question
result = answer_question("What risks should I discuss for a client with high APAC tech exposure?")
print(result["answer"])
```

## Run the eval harness

```bash
python -m eval.run_eval
```

Outputs Recall@K per question plus a full JSON dump to `eval/eval_results.json`.

## Design choices (why, briefly)

| Choice | Reasoning |
|---|---|
| Structure-aware chunking | Fact sheets/policies have real sections; blind fixed-size windows blend unrelated content and hurt citation accuracy |
| Local embeddings by default | No API dependency to get a working prototype; swap to OpenAI in `config.py` for quality |
| Chroma (persistent, local) | Zero server setup, fast enough for hackathon-scale data |
| Structured data kept out of the vector store | Exact numeric/date reasoning needs exact filtering, not similarity search |
| Raw Python, no LangChain | Full visibility for the architecture review and Q&A - every step is one small file |
| Abstention enforced via system prompt phrase | Makes it gradable in the eval harness, not just a vibe |
| Agentic tool-use loop | Model chooses retrieve_documents vs query_portfolio vs both, chains calls for multi-hop questions (bonus: agentic RAG) |
