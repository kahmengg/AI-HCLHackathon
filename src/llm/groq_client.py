import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


_client = None

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_client() -> Groq:
    """
    Create and cache the Groq client.
    """

    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing.\n"
                "Add it to your .env file:\n\n"
                "GROQ_API_KEY=your_real_key_here"
            )

        _client = Groq(api_key=api_key)

    return _client


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send a prompt to Groq and return the generated answer.
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
                    "You are a careful wealth-management assistant. "
                    "Answer only from the supplied evidence. "
                    "Do not invent missing facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.1,

        max_tokens=1200,
    )

    content = response.choices[0].message.content

    if content is None:
        return ""

    return content.strip()


def list_available_models() -> list[str]:
    """
    Return models accessible using the current Groq API key.
    """

    client = get_client()

    response = client.models.list()

    return sorted(
        model.id
        for model in response.data
    )