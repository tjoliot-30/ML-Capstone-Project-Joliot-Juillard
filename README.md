# Operations & Decisions Portfolio — Hospital Pharmacy & Critical Drug Inventory

> **Course:** Advanced Machine Learning — ESILV 2026–2027  
> **Authors:** Axel Juillard & Timothée Joliot  
> **Application Domain:** Healthcare / Hospital Pharmacy Supply Chain (MITM Drugs)  
> **Project Overview:** See [`PROJECT_SUBJECT.md`](PROJECT_SUBJECT.md)

```text
ML-Capstone-Project-Joliot-Juillard/
├── README.md               # Repository overview and metadata
├── PROJECT_SUBJECT.md      # Concise project subject description (English)
├── submission.yaml         # Formal submission metadata
├── claims_evidence.csv     # Scientific claims index (TS-01, RL-01, OPS-01)
├── AI_USE_LOG.md           # Generative AI transparency log
├── CONTRIBUTIONS.csv       # Pair rotating roles log (lead / steward)
├── configs/replay.yaml     # Replay configuration
├── data/README.md          # Data provenance and cryptographic hashes
├── artifacts/              # Frozen deliverables (final_forecast.csv, q_table.npy)
├── evidence/               # Controlled receipts and replay outputs
├── report/release_brief.md # Final synthesis brief (≤ 4 pages)
└── scripts/quick_replay.py # Fail-closed replay runner
```

## Project Summary
This project models the critical supply chain of a Hospital In-House Pharmacy (PUI) managing Major Therapeutic Interest Medicines (MITM). The **Time Series** strand forecasts 28-day antibiotic demand while accounting for censored demand during stockouts (`stockout_flag=1`). The **Reinforcement Learning** strand trains a tabular Q-learning agent to balance cold-storage costs against the severe clinical risks of ICU stockouts.

## Environment & Pair Metadata

- **Authors:** Axel Juillard & Timothée Joliot
- **Operating System:** Windows (x86_64)
- **Python Environment:** Python 3.12.16 locked via `uv`
- **TD1 Status:** Baseline evidence recorded (`TS-01` in `claims_evidence.csv`, lag-7 MAE 9.61 vs. mean 12.56)

## Quick Replay

From the repository root:

```bash
python scripts/quick_replay.py --config configs/replay.yaml
```
