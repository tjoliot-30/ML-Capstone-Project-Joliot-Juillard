# Evidence directory

Store only selected two-strand replay outputs and instructor-returned controlled
artifacts referenced by the claim index. Keep original TD1–TD4/TDo1/TDo2
evidence at its frozen repository path; do not duplicate full notebook trees.

`quick_replay.py` writes `quick_replay_status.json` on every attempt and writes
`quick_replay_receipt.json` only after both declared outputs are freshly created
or refreshed and every declared input remains unchanged.

The `controlled/` directory holds target-free forecast evidence and aggregate
RL evidence. It must never contain protected demand targets, hidden seeds or
seed-level controlled episodes.
