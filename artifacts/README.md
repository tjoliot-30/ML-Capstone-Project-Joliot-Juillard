# Frozen release artifacts

- `final_forecast.csv`: the TDo1 28-day forecast frozen before the protected
  horizon is scored. Preserve its receipt-bound bytes.
- `q_table.npy`: the TDo2 numeric, non-pickle Q-table shaped `(9, 2, 3)` and
  frozen before controlled RL evaluation.

Do not overwrite either artifact after its controlled receipt is issued. A new
artifact requires a new submission ID and whatever resubmission rule ESILV has
formally approved; it is never a silent capstone repair.
