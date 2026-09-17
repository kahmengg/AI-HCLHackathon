from src.structured.structured_data import (
    get_client_profile,
    get_portfolio,
    get_transactions,
    get_potential_suitability_mismatches,
)


def client_profile_tool(client_id: str) -> dict:
    return get_client_profile(client_id)


def portfolio_tool(client_id: str) -> list[dict]:
    return get_portfolio(client_id)


def transactions_tool(client_id: str) -> list[dict]:
    return get_transactions(client_id)


def suitability_mismatches_tool() -> list[dict]:
    return get_potential_suitability_mismatches()