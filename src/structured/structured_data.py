"""
Loads the structured tabular files (clients_portfolio.csv, transactions.csv)
into pandas DataFrames and exposes query functions.

WHY THIS DATA NEVER GOES THROUGH THE CHUNK/EMBED PIPELINE:
Turning "client CL011 holds SGD 480,000 of Fund X, 22% of portfolio" into a
prose chunk and relying on semantic similarity to retrieve it loses exact
filtering and arithmetic (e.g. "clients over the 20% concentration
guideline"). Keeping it as a DataFrame lets the LLM call a tool that filters/
sums exactly, then reasons over the (small, exact) result - instead of hoping
embedding similarity surfaces the right rows.

NOTE ON COLUMN NAMES: the exact column names below are a best guess from the
README ("15 client profiles... risk profile, and full portfolio holdings",
"transactions.csv... 55-row transaction ledger"). Run `inspect_columns()`
once against your real files and adjust CLIENT_ID_COL / VALUE_COL etc. below
if they differ - everything downstream (tools.py, eval) reads through these
constants, so it's a one-place fix.
"""
from pathlib import Path
import pandas as pd

# --- adjust these if the real column names differ ---
CLIENT_ID_COL = "client_id"
HOLDING_VALUE_COL = "market_value"
TXN_DATE_COL = "date"
# -----------------------------------------------------

_portfolio_df: pd.DataFrame | None = None
_transactions_df: pd.DataFrame | None = None


def inspect_columns(data_dir: Path) -> None:
    """Quick sanity check - run this first against the real files and confirm
    CLIENT_ID_COL / HOLDING_VALUE_COL / TXN_DATE_COL above actually exist."""
    portfolio = pd.read_csv(data_dir / "clients_portfolio.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")
    print("clients_portfolio.csv columns:", list(portfolio.columns))
    print("transactions.csv columns:", list(transactions.columns))


def load_structured_data(data_dir: Path):
    global _portfolio_df, _transactions_df
    _portfolio_df = pd.read_csv(data_dir / "clients_portfolio.csv")
    _transactions_df = pd.read_csv(data_dir / "transactions.csv")
    return _portfolio_df, _transactions_df


def get_portfolio(client_id: str) -> pd.DataFrame:
    """All holdings rows for one client."""
    return _portfolio_df[_portfolio_df[CLIENT_ID_COL] == client_id]


def get_transactions(client_id: str, start_date: str | None = None,
                      end_date: str | None = None) -> pd.DataFrame:
    df = _transactions_df[_transactions_df[CLIENT_ID_COL] == client_id]
    if start_date:
        df = df[df[TXN_DATE_COL] >= start_date]
    if end_date:
        df = df[df[TXN_DATE_COL] <= end_date]
    return df


def get_concentration_breaches(threshold_pct: float = 20.0) -> pd.DataFrame:
    """Clients whose single holding exceeds threshold_pct of their total
    portfolio value - directly answers the README's concentration-guideline
    example question."""
    df = _portfolio_df.copy()
    df["pct_of_portfolio"] = df.groupby(CLIENT_ID_COL)[HOLDING_VALUE_COL].transform(
        lambda x: x / x.sum() * 100
    )
    return df[df["pct_of_portfolio"] > threshold_pct]
