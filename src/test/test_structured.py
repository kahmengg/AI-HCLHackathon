from src.structured.structured_data import (
    get_client_profile,
    get_portfolio,
    get_transactions,
    get_concentration_breaches,
    get_potential_suitability_mismatches,
    get_clients_by_risk_profile,
    get_dataset_summary,
)


print("\n==============================")
print("DATABASE SUMMARY")
print("==============================")

print(get_dataset_summary())


print("\n==============================")
print("CL001 PROFILE")
print("==============================")

print(get_client_profile("CL001"))


print("\n==============================")
print("CL001 PORTFOLIO")
print("==============================")

print(get_portfolio("CL001"))


print("\n==============================")
print("CL001 TRANSACTIONS")
print("==============================")

print(get_transactions("CL001"))


print("\n==============================")
print("CONCENTRATION > 20%")
print("==============================")

print(
    get_concentration_breaches(
        threshold_pct=20
    )
)


print("\n==============================")
print("SUITABILITY MISMATCHES")
print("==============================")

print(
    get_potential_suitability_mismatches()
)


print("\n==============================")
print("CONSERVATIVE CLIENTS")
print("==============================")

print(
    get_clients_by_risk_profile(
        "Conservative"
    )
)