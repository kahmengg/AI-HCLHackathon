"""
Central configuration. Keeping every tunable in one file means the eval
harness and pipeline scripts import the same constants - no silent drift
between "what we ingested with" and "what we're evaluating against".
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"          # on-disk vector store
STRUCTURED_CSV_PORTFOLIO = DATA_DIR / "clients_portfolio.csv"
STRUCTURED_CSV_TRANSACTIONS = DATA_DIR / "transactions.csv"

# --- Chunking (word-count based; swap for a tokenizer if you want exact token counts) ---
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 60

# --- Embedding ---
# Local model = no API key needed, works offline. Swap EMBEDDING_BACKEND to
# "openai" + set OPENAI_API_KEY if you want higher-quality embeddings and have
# API access during the hackathon.
EMBEDDING_BACKEND = "local"          # "local" | "openai"
LOCAL_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- Vector store ---
COLLECTION_NAME = (
    "wealth_advisor_documents"
)

