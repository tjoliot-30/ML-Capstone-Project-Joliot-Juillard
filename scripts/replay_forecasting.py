#!/usr/bin/env python3
"""Frozen public forecasting replay (no model fitting, retraining, or tuning)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.evaluation import smape
from shared.forecasting import rolling_origins, seasonal_naive


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen forecasting public replay")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dev_sha = sha256_file(args.dev_csv)
    assert dev_sha == config["dataset_sha256"], f"Dataset SHA256 mismatch: {dev_sha}"

    demand = pd.read_csv(args.dev_csv, parse_dates=["date"])
    assert len(demand) == 1433
    assert demand["date"].is_monotonic_increasing and demand["date"].nunique() == len(demand)
    assert (demand["date"].diff().iloc[1:] == pd.Timedelta(days=1)).all()
    values = demand["demand_units"].to_numpy(float)

    # 1. Public TD1 28-day baseline replay (no fitting)
    train_td1 = values[:-28]
    val_td1 = values[-28:]
    sn_28 = seasonal_naive(train_td1, 28, season=7)
    td1_seasonal_naive = {
        "mae": round(float(mean_absolute_error(val_td1, sn_28)), 6),
        "smape": round(float(smape(val_td1, sn_28)), 6),
    }

    # 2. Public TD2 rolling-origin seasonal-naive comparator replay (no fitting)
    origins = rolling_origins(len(values), initial=1250, horizon=14, step=60, n_origins=3)
    assert list(origins) == config["protocol"]["rolling_origins"]
    sn_folds = []
    for origin in origins:
        pred = seasonal_naive(values[:origin], 14, season=7)
        actual = values[origin : origin + 14]
        sn_folds.append(
            {
                "origin": int(origin),
                "mae": round(float(mean_absolute_error(actual, pred)), 6),
                "smape": round(float(smape(actual, pred)), 6),
            }
        )

    # 3. Validate frozen TDo1 28-day target-free release artifact
    fc_sha = sha256_file(args.forecast)
    assert fc_sha == config["tdo1_release"]["artifact_sha256"], f"Forecast SHA256 mismatch: {fc_sha}"
    fc = pd.read_csv(args.forecast)
    assert len(fc) == 28 and "demand_units" not in fc.columns
    assert list(fc.columns) == ["date", "forecast", "lower_80", "upper_80", "lower_95", "upper_95", "model"]
    assert np.isfinite(fc[["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]].to_numpy()).all()
    assert (fc["forecast"] >= 0).all()
    assert (fc["lower_95"] <= fc["lower_80"]).all()
    assert (fc["lower_80"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["upper_80"]).all()
    assert (fc["upper_80"] <= fc["upper_95"]).all()

    payload = {
        "strand": "forecasting",
        "frozen_replay_only": True,
        "dataset_sha256": dev_sha,
        "td1_28d_seasonal_naive": td1_seasonal_naive,
        "td2_rolling_seasonal_naive": sn_folds,
        "td2_seasonal_naive_mean_mae": round(float(np.mean([f["mae"] for f in sn_folds])), 6),
        "frozen_td2_reference": config["selection_rule"]["dev_results"],
        "tdo1_frozen_release": {
            "model": str(fc["model"].iloc[0]),
            "rows": int(len(fc)),
            "date_min": str(fc["date"].min()),
            "date_max": str(fc["date"].max()),
            "forecast_min": round(float(fc["forecast"].min()), 4),
            "forecast_max": round(float(fc["forecast"].max()), 4),
            "artifact_sha256": fc_sha,
            "nested_bounds_verified": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
