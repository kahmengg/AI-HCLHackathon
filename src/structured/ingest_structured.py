from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

CLIENTS_JSON = DATA_DIR / "clients_portfolio.json"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"


def create_tables(conn):

    conn.executescript(
        """
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS holdings;
        DROP TABLE IF EXISTS clients;

        CREATE TABLE clients (
            client_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            nationality TEXT,
            residency_country TEXT,
            age INTEGER,
            occupation TEXT,
            marital_status TEXT,
            net_worth_band TEXT,
            investor_status TEXT,
            base_currency TEXT,
            aum_sgd REAL,
            risk_profile TEXT,
            risk_score_1_to_10 INTEGER,
            investment_objective TEXT,
            relationship_manager TEXT,
            servicing_branch TEXT,
            kyc_status TEXT,
            pep_status TEXT,
            source_of_wealth TEXT,
            last_portfolio_review_date TEXT,
            suitability_flag TEXT,
            notes TEXT
        );

        CREATE TABLE holdings (
            holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            asset_class TEXT,
            allocation_pct REAL,
            value_sgd REAL,
            currency TEXT,

            FOREIGN KEY (client_id)
                REFERENCES clients(client_id)
        );

        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            date TEXT,
            transaction_type TEXT,
            product_name TEXT,
            amount REAL,
            currency TEXT,
            status TEXT,
            notes TEXT,

            FOREIGN KEY (client_id)
                REFERENCES clients(client_id)
        );

        CREATE INDEX idx_holdings_client
        ON holdings(client_id);

        CREATE INDEX idx_transactions_client
        ON transactions(client_id);

        CREATE INDEX idx_clients_risk
        ON clients(risk_profile);
        """
    )


def load_clients_and_holdings(conn):

    with open(
        CLIENTS_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    clients = data["clients"]

    for client in clients:

        # --------------------------------------
        # Insert client
        # --------------------------------------

        conn.execute(
            """
            INSERT INTO clients (
                client_id,
                name,
                nationality,
                residency_country,
                age,
                occupation,
                marital_status,
                net_worth_band,
                investor_status,
                base_currency,
                aum_sgd,
                risk_profile,
                risk_score_1_to_10,
                investment_objective,
                relationship_manager,
                servicing_branch,
                kyc_status,
                pep_status,
                source_of_wealth,
                last_portfolio_review_date,
                suitability_flag,
                notes
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                client.get("client_id"),
                client.get("name"),
                client.get("nationality"),
                client.get("residency_country"),
                client.get("age"),
                client.get("occupation"),
                client.get("marital_status"),
                client.get("net_worth_band"),
                client.get("investor_status"),
                client.get("base_currency"),
                client.get("aum_sgd"),
                client.get("risk_profile"),
                client.get("risk_score_1_to_10"),
                client.get("investment_objective"),
                client.get("relationship_manager"),
                client.get("servicing_branch"),
                client.get("kyc_status"),
                client.get("pep_status"),
                client.get("source_of_wealth"),
                client.get("last_portfolio_review_date"),
                client.get("suitability_flag"),
                client.get("notes"),
            )
        )

        # --------------------------------------
        # Insert holdings
        # --------------------------------------

        for holding in client.get(
            "portfolio_holdings",
            []
        ):

            conn.execute(
                """
                INSERT INTO holdings (
                    client_id,
                    product_name,
                    asset_class,
                    allocation_pct,
                    value_sgd,
                    currency
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    client["client_id"],
                    holding.get("product_name"),
                    holding.get("asset_class"),
                    holding.get("allocation_pct"),
                    holding.get("value_sgd"),
                    holding.get("currency"),
                )
            )


def load_transactions(conn):

    df = pd.read_csv(TRANSACTIONS_CSV)

    for _, row in df.iterrows():

        conn.execute(
            """
            INSERT INTO transactions (
                transaction_id,
                client_id,
                date,
                transaction_type,
                product_name,
                amount,
                currency,
                status,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["transaction_id"],
                row["client_id"],
                row["date"],
                row["transaction_type"],
                row["product_name"],
                float(row["amount"]),
                row["currency"],
                row["status"],
                row["notes"],
            )
        )


def ingest():

    with get_connection() as conn:

        print("Creating tables...")
        create_tables(conn)

        print("Loading clients and holdings...")
        load_clients_and_holdings(conn)

        print("Loading transactions...")
        load_transactions(conn)

        conn.commit()

        print("Structured data ingestion complete.")


if __name__ == "__main__":
    ingest()