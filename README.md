# Operations & Decisions Portfolio — Hospital Pharmacy & Critical Drug Inventory

> **Course:** Advanced Machine Learning — ESILV 2026–2027  
> **Authors:** Axel Juillard & Timothée Joliot  
> **Application Domain:** Healthcare / Hospital Pharmacy Supply Chain (MITM Drugs)  
> **Project Overview:** See [`PROJECT_SUBJECT.md`](PROJECT_SUBJECT.md)  
> **Release Brief:** See [`report/release_brief.md`](report/release_brief.md)

```text
ML-Capstone-Project-Joliot-Juillard/
├── README.md                        # Repository overview and metadata
├── PROJECT_SUBJECT.md               # Concise project subject description (English)
├── submission.yaml                  # Formal submission metadata
├── claims_evidence.csv              # Scientific claims index (TS-01, RL-01, OPS-01)
├── AI_USE_LOG.md                    # Generative AI transparency log
├── CONTRIBUTIONS.csv                # Pair rotating roles log (lead / steward)
├── 02_labs/                         # Completed student_starter notebooks (TD1–TD4, TDo1–TDo2)
├── shared/                          # Public datasets & shared evaluation primitives (no hidden targets)
├── configs/
│   ├── replay.yaml                  # Two-strand frozen replay configuration
│   ├── td2_forecast_config.yaml     # Frozen TD2/TDo1 forecasting configuration
│   └── td4_rl_config.yaml           # Frozen TD3/TD4/TDo2 RL & safety configuration
├── data/README.md                   # Data provenance and cryptographic hashes
├── artifacts/
│   ├── final_forecast.csv           # Frozen TDo1 28-day forecast (no targets)
│   └── q_table.npy                  # Frozen TD4 Q-table (9, 2, 3)
├── evidence/
│   ├── controlled/                  # Placeholder for instructor-issued receipts (pending)
│   ├── replay/                      # Fresh replay outputs (forecasting & RL)
│   ├── quick_replay_receipt.json    # Fail-closed replay success receipt
│   └── quick_replay_status.json     # Authoritative latest replay attempt status
├── report/release_brief.md          # Final synthesis brief (≤ 4 pages + references)
└── scripts/
    ├── quick_replay.py              # Official fail-closed replay runner
    ├── replay_forecasting.py        # Frozen forecasting public replay script
    └── replay_rl.py                 # Frozen RL safety public replay script
```

## Project Summary
This project models the critical supply chain of a Hospital In-House Pharmacy (PUI) managing Major Therapeutic Interest Medicines (MITM). The **Time Series** strand forecasts 28-day antibiotic demand while accounting for censored demand during stockouts (`stockout_flag=1`). The **Reinforcement Learning** strand trains a tabular Q-learning agent to balance cold-storage costs against the severe clinical risks of ICU stockouts.

## Six-Session & Capstone Status

| Session | Status | Key Outcome & Evidence |
|---|---|---|
| **TD1** | ✅ Completed | Temporal data contract verified (`1433` rows, `b275d65b...`), lag-7 seasonal-naive baseline MAE `9.61` vs mean `12.56`. |
| **TD2** | ✅ Completed | 3-origin rolling comparison (`1250, 1310, 1370`): SARIMA mean MAE `6.14`, ETS `6.17`, seasonal-naive `7.29`; SARIMA frozen under 5% gate. |
| **TD3** | ✅ Completed | MDP & reward decomposition verified (`reward == -cost`); regime-aware reorder cuts cost vs random (`41.16` vs `88.20`) with `99.17%` fill rate. |
| **TD4** | ✅ Completed | Explicit 1,750-episode tabular Q-learning trained; frozen `artifacts/q_table.npy` (`SHA256: 61cce191...`) achieves `42.78` cost and `98.50%` fill rate on 5 disjoint dev seeds. |
| **TDo1** | ✅ Handoff Ready | SARIMA 80%/95% interval coverage (`85.71%` / `100.00%`) across 42 cases; frozen 28-day `artifacts/final_forecast.csv` (`SHA256: 8fe96aba...`) awaiting protected receipt. |
| **TDo2** | ✅ Handoff Ready | Public safety replay (`38.22` cost, `98.50%` fill rate), seed-injected evaluator, and individual transition audit complete; awaiting controlled batch receipt. |
| **Capstone** | ✅ Pre-Receipt Ready | Two-strand `quick_replay.py` passed (`5.55s`, receipt `b8e38bf5...`); `report/release_brief.md` complete with 3 verified papers (`HOLD` pending controlled receipts). |

## Environment & Pair Metadata

- **Team:** Axel Juillard (`ajuillard`) & Timothée Joliot (`tjoliot`)
- **Tested OS / Python / Lock:** Windows / Python 3.12.16 / `uv.lock`
- **Reference Hardware:** AMD Ryzen AI 7 350 w/ Radeon 860M (16 logical cores, 31.1 GiB RAM, CPU-only)
- **Two-Strand Replay Runtime:** `5.55 seconds` (< 1 minute; limit 10 min)
- **Replay Outputs:** `evidence/replay/forecasting/result.json` & `evidence/replay/reinforcement_learning/result.json`
- **Known Deviation:** Controlled receipts from TDo1/TDo2 are pending instructor return (`partial` status on `TS-01`, `RL-01`, `OPS-01`).

## Quick Replay

From the repository root:

```bash
python scripts/quick_replay.py --config configs/replay.yaml
```
