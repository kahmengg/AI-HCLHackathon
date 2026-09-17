import os

from dotenv import load_dotenv
from groq import Groq


# Load variables from .env
load_dotenv()


_client = None


DEFAULT_MODEL = "qwen/qwen3.6-27b"


def get_client() -> Groq:
    """
    Create and cache the Groq client.

    GROQ_API_KEY is read from the local .env file.
    """

    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing.\n"
                "Add it to your local .env file:\n\n"
                "GROQ_API_KEY=your_real_key_here"
            )

        _client = Groq(
            api_key=api_key
        )

    return _client


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send a prompt to Groq and return the generated text.

    This wrapper is deliberately small so the rest of the
    RAG system does not depend directly on the Groq SDK.
    """

    if not prompt.strip():
        return ""

    client = get_client()

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful wealth-management "
                    "assistant. Follow the supplied evidence "
                    "and instructions exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        # Qwen 3.6 supports reasoning_effort='none'
        # for normal/faster dialogue.
        reasoning_effort="none",

        temperature=0.2,

        max_tokens=1200,
    )

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content.strip()


def list_available_models() -> list[str]:
    """
    Return the Groq models available to the current API key.

    Useful when a model ID changes or access differs by account.
    """

    client = get_client()

    response = client.models.list()

    return sorted(
        model.id
        for model in response.data
    )