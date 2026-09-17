"""
Loads client_correspondence.json ONLY - this is the one JSON file that
contains unstructured prose (email bodies) and belongs in the vector store.

clients_portfolio.json is structured client/holdings data and is handled by
src/structured/structured_data.py instead - it should never be chunked or
embedded, see that module's docstring for why.

NOTE: the exact key names below (thread_id, client_id, messages, body) are
a best guess based on the README description ("7 email threads (client<->RM
and internal Compliance<->RM)"). Open the real file once and adjust the
`.get(...)` keys in `_iter_messages` if the schema differs - everything else
in the pipeline is unaffected by this.
"""
import json
from pathlib import Path


def _iter_messages(thread: dict):
    """Yield (message_dict) for a single thread, tolerant of a few likely
    schema variants (messages/emails, body/text/content)."""
    messages = thread.get("messages") or thread.get("emails") or []
    for msg in messages:
        yield msg


def load_correspondence(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tolerate either a top-level list of threads or {"threads": [...]}
    threads = data if isinstance(data, list) else data.get("threads", [])

    records = []
    for thread in threads:
        thread_id = thread.get("thread_id", thread.get("id", "unknown_thread"))
        client_id = thread.get("client_id", "n/a")

        for msg in _iter_messages(thread):
            body = msg.get("body") or msg.get("text") or msg.get("content") or ""
            if not body.strip():
                continue
            sender = msg.get("from", msg.get("sender", "unknown"))
            date = msg.get("date", "unknown")
            records.append({
                "source": f"{path.name}#{thread_id}",
                "doc_type": "correspondence",
                "client_id": client_id,
                # Keep sender/date in the text itself so a retrieved chunk is
                # self-contained enough to cite without a second lookup.
                "text": f"From: {sender} | Date: {date}\n{body}",
            })
    return records
