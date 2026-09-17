from pathlib import Path
import sqlite3


# structured/database.py
#
# __file__ =
# HCLHackathon/src/structured/database.py
#
# parents[0] = structured
# parents[1] = src
# parents[2] = HCLHackathon

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

DB_PATH = DATA_DIR / "wealth_advisor.db"


def get_connection():
    """
    Open a connection to the Wealth Advisor SQLite database.
    """

    conn = sqlite3.connect(DB_PATH)

    # Allows:
    # row["client_id"]
    #
    # instead of only:
    # row[0]
    conn.row_factory = sqlite3.Row

    # Enable SQLite foreign-key checking
    conn.execute("PRAGMA foreign_keys = ON")

    return conn