import pandas as pd

from src.structured.database import get_connection


# ==========================================================
# CLIENT
# ==========================================================

def get_client_profile(
    client_id: str
) -> dict | None:

    client_id = client_id.upper().strip()

    with get_connection() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM clients
            WHERE client_id = ?
            """,
            (client_id,)
        ).fetchone()

    if row is None:
        return None

    return dict(row)


# ==========================================================
# PORTFOLIO
# ==========================================================

def get_portfolio(
    client_id: str
) -> pd.DataFrame:

    client_id = client_id.upper().strip()

    with get_connection() as conn:

        query = """
        SELECT
            holding_id,
            client_id,
            product_name,
            asset_class,
            allocation_pct,
            value_sgd,
            currency
        FROM holdings
        WHERE client_id = ?
        ORDER BY allocation_pct DESC
        """

        return pd.read_sql_query(
            query,
            conn,
            params=(client_id,)
        )


# ==========================================================
# TRANSACTIONS
# ==========================================================

def get_transactions(
    client_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:

    client_id = client_id.upper().strip()

    query = """
    SELECT *
    FROM transactions
    WHERE client_id = ?
    """

    params = [client_id]

    if start_date:

        query += """
        AND date >= ?
        """

        params.append(start_date)

    if end_date:

        query += """
        AND date <= ?
        """

        params.append(end_date)

    query += """
    ORDER BY date DESC
    """

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn,
            params=params
        )


# ==========================================================
# CONCENTRATION
# ==========================================================

def get_concentration_breaches(
    threshold_pct: float = 20.0,
    include_equal: bool = False,
) -> pd.DataFrame:

    operator = ">=" if include_equal else ">"

    query = f"""
    SELECT
        c.client_id,
        c.name,
        c.risk_profile,
        c.risk_score_1_to_10,

        h.product_name,
        h.asset_class,
        h.allocation_pct,
        h.value_sgd

    FROM clients c

    JOIN holdings h
        ON c.client_id = h.client_id

    WHERE h.allocation_pct {operator} ?

    ORDER BY h.allocation_pct DESC
    """

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn,
            params=(threshold_pct,)
        )


# ==========================================================
# RISK PROFILE
# ==========================================================

def get_clients_by_risk_profile(
    risk_profile: str
) -> pd.DataFrame:

    query = """
    SELECT
        client_id,
        name,
        age,
        risk_profile,
        risk_score_1_to_10,
        aum_sgd,
        investor_status,
        relationship_manager

    FROM clients

    WHERE LOWER(risk_profile) = LOWER(?)

    ORDER BY risk_score_1_to_10 DESC
    """

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn,
            params=(risk_profile,)
        )


# ==========================================================
# RISK SCORE
# ==========================================================

def get_clients_by_risk_score(
    minimum_score: int | None = None,
    maximum_score: int | None = None,
) -> pd.DataFrame:

    query = """
    SELECT
        client_id,
        name,
        risk_profile,
        risk_score_1_to_10,
        aum_sgd,
        investor_status

    FROM clients

    WHERE 1 = 1
    """

    params = []

    if minimum_score is not None:

        query += """
        AND risk_score_1_to_10 >= ?
        """

        params.append(minimum_score)

    if maximum_score is not None:

        query += """
        AND risk_score_1_to_10 <= ?
        """

        params.append(maximum_score)

    query += """
    ORDER BY risk_score_1_to_10 DESC
    """

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn,
            params=params
        )


# ==========================================================
# SUITABILITY
# ==========================================================

def get_potential_suitability_mismatches() -> pd.DataFrame:

    query = """
    SELECT
        client_id,
        name,
        risk_profile,
        risk_score_1_to_10,
        aum_sgd,
        suitability_flag,
        relationship_manager

    FROM clients

    WHERE suitability_flag LIKE 'POTENTIAL MISMATCH%'

    ORDER BY client_id
    """

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn
        )


# ==========================================================
# PRODUCT SEARCH
# ==========================================================

def get_clients_holding_product(
    product_name: str
) -> pd.DataFrame:

    query = """
    SELECT
        c.client_id,
        c.name,
        c.risk_profile,

        h.product_name,
        h.asset_class,
        h.allocation_pct,
        h.value_sgd,

        c.suitability_flag

    FROM clients c

    JOIN holdings h
        ON c.client_id = h.client_id

    WHERE LOWER(h.product_name)
        LIKE LOWER(?)

    ORDER BY h.allocation_pct DESC
    """

    search = f"%{product_name}%"

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn,
            params=(search,)
        )


# ==========================================================
# ASSET CLASS SEARCH
# ==========================================================

def get_clients_holding_asset_class(
    asset_class: str
) -> pd.DataFrame:

    query = """
    SELECT
        c.client_id,
        c.name,
        c.risk_profile,

        h.product_name,
        h.asset_class,
        h.allocation_pct,
        h.value_sgd

    FROM clients c

    JOIN holdings h
        ON c.client_id = h.client_id

    WHERE LOWER(h.asset_class)
        LIKE LOWER(?)

    ORDER BY h.allocation_pct DESC
    """

    search = f"%{asset_class}%"

    with get_connection() as conn:

        return pd.read_sql_query(
            query,
            conn,
            params=(search,)
        )


# ==========================================================
# DATASET SUMMARY
# ==========================================================

def get_dataset_summary() -> dict:

    with get_connection() as conn:

        client_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM clients
            """
        ).fetchone()[0]

        holding_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM holdings
            """
        ).fetchone()[0]

        transaction_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM transactions
            """
        ).fetchone()[0]

    return {
        "clients": client_count,
        "holdings": holding_count,
        "transactions": transaction_count,
    }