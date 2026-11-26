# Operations & Decisions Portfolio — Hospital Pharmacy & Critical Drug Inventory

> **Course:** Advanced Machine Learning — ESILV 2026–2027  
> **Authors:** Axel Juillard & Timothée Joliot  
> **Application Domain:** Healthcare / Hospital Pharmacy Supply Chain (MITM Drugs)  
> **Project Overview:** See [`PROJECT_SUBJECT.md`](PROJECT_SUBJECT.md)

```text
ML-Capstone-Project-Joliot-Juillard/
├── README.md                        # Repository overview and metadata
├── PROJECT_SUBJECT.md               # Concise project subject description (English)
├── submission.yaml                  # Formal submission metadata
├── claims_evidence.csv              # Scientific claims index (TS-01, RL-01, OPS-01)
├── AI_USE_LOG.md                    # Generative AI transparency log
├── CONTRIBUTIONS.csv                # Pair rotating roles log (lead / steward)
├── 02_labs/                         # Completed student_starter notebooks (TD1–TD4, TDo1–TDo2)
├── configs/
│   ├── replay.yaml                  # Replay configuration (to be completed at Capstone)
│   ├── td2_forecast_config.yaml     # Frozen TD2/TDo1 forecasting configuration
│   └── td4_rl_config.yaml           # Frozen TD3/TD4/TDo2 RL & safety configuration
├── data/README.md                   # Data provenance and cryptographic hashes
├── artifacts/
│   ├── final_forecast.csv           # Frozen TDo1 28-day forecast (no targets)
│   └── q_table.npy                  # Frozen TD4 Q-table (9, 2, 3)
├── evidence/                        # Controlled receipts and replay outputs
├── report/release_brief.md          # Final synthesis brief (≤ 4 pages)
└── scripts/quick_replay.py          # Fail-closed replay runner
```

## Project Summary
This project models the critical supply chain of a Hospital In-House Pharmacy (PUI) managing Major Therapeutic Interest Medicines (MITM). The **Time Series** strand forecasts 28-day antibiotic demand while accounting for censored demand during stockouts (`stockout_flag=1`). The **Reinforcement Learning** strand trains a tabular Q-learning agent to balance cold-storage costs against the severe clinical risks of ICU stockouts.

## Six-Session Progress (TD1 – TDo2)

| Session | Status | Key Outcome & Evidence |
|---|---|---|
| **TD1** | ✅ Completed | Temporal data contract verified (`1433` rows, `b275d65b...`), lag-7 seasonal-naive baseline MAE `9.61` vs mean `12.56`. |
| **TD2** | ✅ Completed | 3-origin rolling comparison (`1250, 1310, 1370`): SARIMA mean MAE `6.14`, ETS `6.17`, seasonal-naive `7.29`; SARIMA frozen under 5% gate. |
| **TD3** | ✅ Completed | MDP & reward decomposition verified (`reward == -cost`); regime-aware reorder cuts cost vs random (`41.16` vs `88.20`) with `99.17%` fill rate. |
| **TD4** | ✅ Completed | Explicit 1,750-episode tabular Q-learning trained; frozen `artifacts/q_table.npy` (`SHA256: 61cce191...`) achieves `42.78` cost and `98.50%` fill rate on 5 disjoint dev seeds. |
| **TDo1** | ✅ Handoff Ready | SARIMA 80%/95% interval coverage (`85.71%` / `100.00%`) across 42 cases; frozen 28-day `artifacts/final_forecast.csv` (`SHA256: 8fe96aba...`) awaiting protected receipt. |
| **TDo2** | ✅ Handoff Ready | Public safety replay (`38.22` cost, `98.50%` fill rate), seed-injected evaluator, and `OPS-01` (`HOLD` pending controlled receipts) prepared. |

## Environment & Pair Metadata

- **Authors:** Axel Juillard & Timothée Joliot (alternating `experiment_lead` & `evidence_steward` across TD1–TDo2)
- **Operating System:** Windows (x86_64)
- **Python Environment:** Python 3.12.16 locked via `uv` (`pyproject.toml` / `uv.lock`)

## Quick Replay

From the repository root (once configured for Capstone finalization):

```bash
python scripts/quick_replay.py --config configs/replay.yaml
```
