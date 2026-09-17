# Wealth Management AI Assistant — Sample Dataset (APAC)

A complete sample dataset for building/testing a Wealth Management RAG assistant.
The people, accounts, and institutions represented — including "Meridian Peak
Wealth Partners" — are fictitious and prepared for illustrative and training
purposes.

## Why prepared, not sourced

Real client portfolios, KYC files, and internal policy documents are
confidential and cannot be sourced from the internet. Real fund fact sheets
exist publicly but are copyrighted issuer documents. A prepared dataset is
the standard approach for this kind of project — it also lets you control
ground truth (e.g. exactly which client/product pairs are "unsafe") so you
can score retrieval and reasoning quality objectively.

## Files

### Client & transaction data
| File | Contents |
|---|---|
| `clients_portfolio.json` | 15 client profiles (Singapore, Hong Kong, India, Indonesia, Malaysia, Thailand, Philippines, Vietnam, Japan, South Korea, Taiwan, Australia, Mainland China) with KYC attributes, risk profile, and full portfolio holdings |
| `clients_portfolio.csv` | Same data flattened to one row per holding (70 rows) |
| `transactions.csv` | 55-row transaction ledger (subscriptions, redemptions, coupons, top-ups, pending items) across all 15 clients, Jan–Aug 2026 |

### Policy documents
| File | Contents |
|---|---|
| `policy_kyc_onboarding.pdf` | KYC & Client Onboarding Policy (MAS SFA 04-N02-style, PEP screening, cross-border booking rules) |
| `policy_investment_suitability.pdf` | Investment Suitability, Risk Profiling & Complex Product Due Diligence Policy (FAA-N16-style) |

### Product fact sheets (10 total)
| File | Product | SRI |
|---|---|---|
| `fund_factsheet_safe.pdf` | APAC Stable Income Money Market Fund | 1/7 |
| `fund_factsheet_global_bond_income.pdf` | Global Bond Income Fund | 2/7 |
| `fund_factsheet_global_reit_basket.pdf` | Global Real Estate Income Trust Basket | 3/7 |
| `fund_factsheet_balanced_income_growth.pdf` | Balanced Income & Growth Fund | 3/7 |
| `fund_factsheet_pacific_growth_equity.pdf` | Pacific Growth Equity Fund | 4/7 |
| `fund_factsheet_ilp_regular_premium.pdf` | Meridian Peak Regular Premium Investment-Linked Policy (insurance-linked) | 4/7 |
| `fund_factsheet_global_tech_innovation.pdf` | Global Technology Innovation Fund | 5/7 |
| `fund_factsheet_dci_aud_usd.pdf` | Dual Currency Investment (DCI) — AUD/USD (FX-linked, Complex/SIP) | 5/7 |
| `fund_factsheet_private_equity_fund_iv.pdf` | Private Equity Co-Investment Vehicle – Fund IV (Complex/illiquid) | 6/7 |
| `fund_factsheet_exotic_unsafe.pdf` | APEX Global Multi-Asset Autocallable Note Series 7 (Complex/SIP) | 7/7 |

### Operational / unstructured documents
| File | Contents |
|---|---|
| `rm_call_notes_log.pdf` | 10 RM call/meeting notes across clients, Jan–Aug 2026, including comprehension and concentration flags |
| `client_correspondence.json` | 7 email threads (client↔RM and internal Compliance↔RM), including the original CL013 de-risking email and two internal compliance-escalation threads |
| `client_complaint_letters.pdf` | 2 formal complaint letters (CL002 mis-sale; CL013 delayed rebalancing) with register/status metadata |
| `complex_product_risk_acknowledgement_forms.pdf` | Complex Product Risk Acknowledgement Form — one completed example (CL001) + blank template |

### Evaluation
| File | Contents |
|---|---|
| `golden_qa_dataset.json` | 5 golden question/answer pairs spanning simple lookup, suitability breach, numeric/temporal reasoning, ambiguous policy interpretation, and multi-hop timeline reconstruction, each with expected answer, required source documents, and grading notes |

## Suggested use in your RAG system

1. **Ingest** all PDFs and the `client_correspondence.json` emails as your document corpus.
2. **Ingest** `clients_portfolio.json` / `transactions.csv` as metadata, SQL rows, or per-record text chunks.
3. **Evaluate** against `golden_qa_dataset.json` to score both retrieval (right source documents) and generation (correct, appropriately calibrated answer).
4. **Test retrieval + reasoning** across documents, e.g.:
   - *"Is the APEX Autocallable Note suitable for a Conservative client, and does Robert Chua have a signed risk acknowledgement?"* → needs the suitability policy + fact sheet + the (missing) acknowledgement form + CL002's record.
   - *"Walk me through what happened with Carlos Bautista's DCI trade."* → needs the call note, transaction ledger, internal compliance email, and client record to reconstruct the timeline.
   - *"Has James Sullivan's portfolio been rebalanced since he asked to de-risk?"* → needs the original email, the call note, the complaint letter, and the transaction ledger's "Pending" entry — all should agree nothing has been executed yet.
   - *"Which clients hold Complex Products above the 20% concentration guideline, and does the guideline apply to Accredited Investors?"* → tests policy-text interpretation against portfolio data (CL011 edge case).

## Reference scenarios built into the data

- **CL002 (Robert Chua)** — Conservative client, structured note with no signed acknowledgement, documented comprehension issue, and now a formal complaint. Fully traceable across the call note, internal compliance email, complaint letter, and transaction ledger.
- **CL008 (Carlos Bautista)** — Retail investor sold an FX-linked SIP (DCI) with no CAR/CKA on file — a subtler variant of the CL002 issue, using a different product type. Traceable across the call note, internal compliance email, and transaction ledger (including a realised FX loss on one rollover).
- **CL011 (Park Ji-hoon)** — Accredited Investor with 35% concentration in one Complex Product. Genuinely ambiguous: the 20% cap in `policy_investment_suitability.pdf` Section 4 is worded as applying to Retail Investors specifically — good test of whether your RAG system over- or under-flags.
- **CL013 (James Sullivan)** — Profile changed from Growth to Conservative on notice of retirement; portfolio not yet rebalanced. Traceable across his original email, the follow-up call note, his complaint letter, and the "Pending" transaction ledger entry.
- **CL014 (Zhang Wei)** — PEP-adjacent monitoring case; tests correct application of the KYC policy's PEP provisions to a "cleared but monitored" status rather than a simple yes/no PEP flag.
- **CL015 (Arjun Mehta)** — Cross-border LRS (India remittance scheme) headroom check; tests numerical reasoning across an email thread and the client record.
- **CL001 / CL009** — Clean "positive control" cases: Complex Products held with signed acknowledgements and profiles that genuinely match, for contrast against the flagged cases above.

## Important note on this dataset

Every name, account, institution, and event in this package — including
Meridian Peak Wealth Partners, its staff, and all client records — is
fictitious and was prepared for illustrative and training purposes. Any
resemblance to actual persons, entities, or events is coincidental. This
material must not be used, presented, or circulated as real financial,
legal, or compliance advice, or as evidence of any real institution's
actual policies or dealings.
