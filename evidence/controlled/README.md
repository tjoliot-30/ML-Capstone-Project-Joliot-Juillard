# Instructor-returned controlled evidence

Expected student-visible files after the official handoffs:

- `forecast_holdout_receipt.json` — target-free protected-horizon aggregate;
- `rl_hidden_evaluation_aggregate.csv` and detached SHA256;
- `rl_hidden_evaluation_receipt.yaml` and detached SHA256.

Verify each returned hash and map its evidence ID in `submission.yaml` and the
corresponding claim row. Never add protected demand targets, hidden RL seeds or
seed-level controlled episodes to this directory.
