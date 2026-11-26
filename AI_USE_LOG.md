# Generative-AI Use Log

> If no generative AI was used, write `No generative AI used` in the declaration
> and leave the activity tables empty. Never paste personal, confidential, or
> restricted data here.

## Declaration

**Team:** `Axel Juillard & Timothée Joliot`  
**Generative AI used:** `yes`  
**Declaration:** `We used an AI coding assistant as an engineering aid for code scaffolding, notebook execution checks, and documentation formatting across TD1–TD4 and TDo1–TDo2. We reviewed, executed, and verified every retained output, metric, and interpretation ourselves.`

We understand that AI output is neither experimental evidence nor a scientific
source. We verified every retained contribution against submitted code,
configuration, data provenance, primary sources, and result artifacts.

## Material-use record

| ID | Date | Tool/model if visible | Purpose | Input-context description | Output retained or rejected | Verification performed | Affected artifact paths | Student(s) responsible |
|---|---|---|---|---|---|---|---|---|
| AI-01 | `2026-11-20` | `AI Coding Assistant` | `TD1 data contract invariants, baseline implementation, repository setup` | `Public daily_demand_development.csv schema and TD1 starter notebook prompts (no protected targets)` | `Retained assert_daily_contract, baseline_forecasts, and PROJECT_SUBJECT.md draft after shortening` | `Executed top-to-bottom via nbclient on Python 3.12; verified negative test and MAE/sMAPE values` | `02_labs/TD1_temporal_data_baselines/, PROJECT_SUBJECT.md` | `Timothée Joliot & Axel Juillard` |
| AI-02 | `2026-11-26` | `AI Coding Assistant` | `TD2–TD4 notebook TODO completion, config archiving, and claim traceability` | `Public TD2–TD4 starter notebooks, InventoryControlEnv contract, and public seeds (no protected targets or hidden seeds)` | `Retained rolling comparison loop, MDP verification, explicit Q-learning loop, and frozen config YAMLs` | `Executed all 6 TD2–TD4 notebooks top-to-bottom with exit code 0; verified q_table.npy shape (9,2,3) and SHA256` | `02_labs/TD2..TD4/, configs/td2_forecast_config.yaml, configs/td4_rl_config.yaml, artifacts/q_table.npy, claims_evidence.csv` | `Axel Juillard & Timothée Joliot` |
| AI-03 | `2026-11-26` | `AI Coding Assistant` | `TDo1 interval/forecast release and TDo2 seed-injected evaluator, safety gate & OPS-01 synthesis` | `Public TDo1–TDo2 starter notebooks, future descriptors without targets, and public safety seeds (200011..200071)` | `Retained interval ledger, final_forecast.csv export, compare_on_injected_seeds, individual audit, and OPS-01 claim` | `Executed all 4 TDo1–TDo2 notebooks top-to-bottom; verified nested bounds, zero demand_units leakage, and SHA256 hashes` | `02_labs/TDo1..TDo2/, artifacts/final_forecast.csv, claims_evidence.csv, submission.yaml` | `Timothée Joliot & Axel Juillard` |

## Corrections and rejected suggestions

When material AI assistance was used, record at least one correction or rejected
suggestion. This shows verification; it is not a penalty.

| AI ID | Problem found | How it was detected | Correction or decision |
|---|---|---|---|
| AI-01 | `Initial PROJECT_SUBJECT.md draft was overly verbose and in French instead of concise English` | `Pair review of repository documentation clarity` | `Rejected verbose draft; rewrote PROJECT_SUBJECT.md into 4 concise English bullet points` |
| AI-02 | `Potential temptation to tune Q-learning hyperparameters when it trailed reorder (42.78 vs 42.48)` | `Verification against TD4 Assessed Core protocol (fixed alpha=0.18, gamma=0.96, 350 episodes/seed)` | `Kept exact assessed hyperparameters unchanged and documented the honest comparison and boundary defect at (stock=2, regime=1)` |
| AI-03 | `Controlled instructor receipts are not available prior to instructor submission` | `Checking evidence/controlled/ against TDo1/TDo2 rubric rules` | `Kept controlled_aggregate empty and marked TS-01, RL-01, and OPS-01 as partial pending_instructor_receipt without fabricating receipts` |
