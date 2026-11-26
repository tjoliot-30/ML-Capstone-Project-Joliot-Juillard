#!/usr/bin/env python3
"""Frozen public RL safety replay (no policy training, tuning, or sweeping)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.evaluation import mean_ci95
from shared.rl import evaluate_policies, q_table_policy, random_policy, reorder_policy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_block(df):
    summary = {}
    for policy, grp in df.groupby("policy"):
        c_m, c_l, c_h = mean_ci95(grp["episode_cost"])
        f_m, f_l, f_h = mean_ci95(grp["mean_fill_rate"])
        summary[str(policy)] = {
            "episodes": int(len(grp)),
            "cost_mean": round(c_m, 4),
            "cost_ci95": [round(c_l, 4), round(c_h, 4)],
            "fill_rate_mean": round(f_m, 4),
            "fill_rate_ci95": [round(f_l, 4), round(f_h, 4)],
            "unmet_demand_mean": round(float(grp["unmet_demand"].mean()), 4),
            "mean_stock": round(float(grp["mean_stock"].mean()), 4),
            "mean_order_quantity": round(float(grp["mean_order_quantity"].mean()), 4),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen RL public safety replay")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--q-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    qt_sha = sha256_file(args.q_table)
    assert qt_sha == config["frozen_artifact"]["sha256"], f"Q-table SHA256 mismatch: {qt_sha}"

    q_table = np.load(args.q_table, allow_pickle=False)
    assert q_table.shape == (9, 2, 3) and np.isfinite(q_table).all()

    policies = {
        "random": random_policy,
        "reorder": reorder_policy,
        "q_learning": q_table_policy(q_table),
    }

    dev_seeds = config["development_evaluation"]["disjoint_seeds"]
    safety_seeds = config["tdo2_public_safety"]["disjoint_seeds"]

    dev_df = evaluate_policies(policies, seeds=dev_seeds)
    safety_df = evaluate_policies(policies, seeds=safety_seeds)

    dev_summary = summarize_block(dev_df)
    safety_summary = summarize_block(safety_df)

    q_safe = safety_summary["q_learning"]
    r_safe = safety_summary["reorder"]
    service_floor_met = q_safe["fill_rate_mean"] >= config["tdo2_public_safety"]["service_floor"]
    cost_bound_met = q_safe["cost_mean"] <= config["tdo2_public_safety"]["cost_bound_vs_reorder"] * r_safe["cost_mean"]

    payload = {
        "strand": "reinforcement_learning",
        "frozen_replay_only": True,
        "q_table_shape": list(q_table.shape),
        "q_table_sha256": qt_sha,
        "td4_development_replay": {
            "seeds": dev_seeds,
            "summary": dev_summary,
        },
        "tdo2_public_safety_replay": {
            "seeds": safety_seeds,
            "summary": safety_summary,
            "service_floor_met": bool(service_floor_met),
            "cost_within_105pct_reorder": bool(cost_bound_met),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
