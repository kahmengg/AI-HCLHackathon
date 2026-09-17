"""
Load client_correspondence.json.

This JSON contains unstructured email text, so it belongs in the
RAG/vector-store pipeline.

clients_portfolio.json is structured data and should be handled
separately through SQLite.
"""

import json
from pathlib import Path


def load_correspondence(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Correspondence file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    threads = data.get("email_threads", [])

    records = []

    for thread in threads:

        thread_id = thread.get(
            "thread_id",
            "unknown_thread"
        )

        subject = thread.get(
            "subject",
            "No subject"
        )

        client_id = thread.get(
            "related_client_id",
            "n/a"
        )

        messages = thread.get(
            "messages",
            []
        )

        for message_index, msg in enumerate(messages):

            body = msg.get("body", "").strip()

            if not body:
                continue

            sender = msg.get(
                "from",
                "unknown"
            )

            recipient = msg.get(
                "to",
                "unknown"
            )

            date = msg.get(
                "date",
                "unknown"
            )

            text = f"""
Subject: {subject}
From: {sender}
To: {recipient}
Date: {date}

{body}
""".strip()

            records.append(
                {
                    "source": path.name,
                    "doc_type": "correspondence",
                    "client_id": client_id,
                    "thread_id": thread_id,
                    "subject": subject,
                    "message_index": message_index,
                    "date": date,
                    "text": text,
                }
            )

    return records