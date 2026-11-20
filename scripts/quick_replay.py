#!/usr/bin/env python3
"""Run exactly two frozen capstone replays and emit a fail-closed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STRANDS = ["forecasting", "reinforcement_learning"]
MAX_TOTAL_SECONDS = 600.0
FILESYSTEM_MTIME_TOLERANCE_NS = 2_000_000_000
FORBIDDEN_COMMAND_PARTS = {
    "fit",
    "fitting",
    "gridsearch",
    "hyperparameter",
    "hyperparameters",
    "optimize",
    "optimization",
    "optimise",
    "optimisation",
    "randomsearch",
    "retrain",
    "retraining",
    "sweep",
    "sweeps",
    "train",
    "training",
    "tune",
    "tuning",
}
PATH_SUFFIXES = {
    ".csv",
    ".json",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Do not follow a pre-planted temporary-file symlink outside the repository.
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def within_root(root: Path, raw_path: str, *, label: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be repository-relative: {raw_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository root: {raw_path}") from exc
    return resolved


def required_input(root: Path, raw_path: str, *, label: str) -> Path:
    path = within_root(root, raw_path, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {raw_path}")
    return path


def relative_record(root: Path, path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": stat.st_size,
    }


def output_snapshot(root: Path, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": path.relative_to(root).as_posix(), "exists": False}
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected output must be a regular non-symlink file: {path.relative_to(root)}")
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "exists": True,
        "sha256": sha256(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def changed_since(before: dict[str, Any], after: dict[str, Any], started_wall_ns: int) -> bool:
    if not after.get("exists"):
        return False
    # Some supported filesystems expose timestamps at one-second granularity.
    # A small tolerance avoids rejecting a genuinely new file created in the
    # same coarse clock tick, while the before/after mutation requirement still
    # rejects a pre-existing untouched output.
    if int(after["mtime_ns"]) < started_wall_ns - FILESYSTEM_MTIME_TOLERANCE_NS:
        return False
    if not before.get("exists"):
        return True
    return any(
        after.get(field) != before.get(field)
        for field in ("sha256", "bytes", "mtime_ns", "inode")
    )


def require_unchanged(before: list[dict[str, Any]], after: list[dict[str, Any]], *, label: str) -> None:
    before_hashes = [(record["path"], record["sha256"]) for record in before]
    after_hashes = [(record["path"], record["sha256"]) for record in after]
    if before_hashes != after_hashes:
        raise RuntimeError(f"Declared {label} changed during replay")


def token_parts(token: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", token.lower()) if part}


def validate_command_tokens(
    root: Path,
    argv: list[str],
    script_path: str,
    declared_paths: set[Path],
) -> list[str]:
    if len(argv) < 2 or Path(argv[0]).name not in {"python", "python3"}:
        raise ValueError("Each replay command must start with python or python3 and a script path")
    if argv[1] != script_path:
        raise ValueError("argv[1] must exactly match script_path")
    forbidden = sorted({part for token in argv for part in token_parts(token)} & FORBIDDEN_COMMAND_PARTS)
    if forbidden:
        raise ValueError(f"Training/tuning command token rejected: {', '.join(forbidden)}")

    for token in argv[1:]:
        value = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
        if value.startswith("-") and value == token:
            continue
        if "/" in value or "\\" in value or Path(value).suffix.lower() in PATH_SUFFIXES:
            resolved = within_root(root, value, label="argv path")
            if resolved not in declared_paths:
                raise ValueError(f"argv path is not declared for hashing: {value}")
    return [sys.executable, *argv[1:]]


def string_list(item: dict[str, Any], key: str, *, strand: str) -> list[str]:
    value = item.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{strand}.{key} must be a non-empty list of repository-relative paths")
    return value


def validate_item(root: Path, item: dict[str, Any], expected_strand: str) -> dict[str, Any]:
    if item.get("strand") != expected_strand:
        raise ValueError(f"Expected strand {expected_strand}")
    if item.get("frozen_replay_only") is not True or item.get("changes_model_or_protocol") is not False:
        raise ValueError(f"Replay freeze declaration is missing or unsafe for {expected_strand}")

    argv = item.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(token, str) and token for token in argv):
        raise ValueError(f"Invalid argv for {expected_strand}")
    script_path = item.get("script_path")
    if not isinstance(script_path, str) or not script_path:
        raise ValueError(f"{expected_strand}.script_path is required")
    script = required_input(root, script_path, label=f"{expected_strand} script")
    if script.suffix != ".py":
        raise ValueError(f"{expected_strand}.script_path must name a Python file")

    configuration_paths = string_list(item, "configuration_inputs", strand=expected_strand)
    data_paths = string_list(item, "data_inputs", strand=expected_strand)
    configuration_inputs = [
        required_input(root, path, label=f"{expected_strand} configuration input")
        for path in configuration_paths
    ]
    data_inputs = [
        required_input(root, path, label=f"{expected_strand} data input")
        for path in data_paths
    ]

    expected_output_raw = item.get("expected_output")
    if not isinstance(expected_output_raw, str) or not expected_output_raw:
        raise ValueError(f"{expected_strand}.expected_output is required")
    lexical_output = root / Path(expected_output_raw)
    if lexical_output.is_symlink():
        raise ValueError(f"{expected_strand} expected output must not be a symlink")
    expected_output = within_root(root, expected_output_raw, label=f"{expected_strand} expected output")

    return {
        "strand": expected_strand,
        "declared_argv": argv,
        "executed_argv": validate_command_tokens(
            root,
            argv,
            script_path,
            {script, *configuration_inputs, *data_inputs, expected_output},
        ),
        "script": script,
        "configuration_inputs": configuration_inputs,
        "data_inputs": data_inputs,
        "expected_output": expected_output,
    }


def run_replays(config_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    evidence_dir = within_root(root, "evidence", label="internal evidence directory")
    receipt_path = evidence_dir / "quick_replay_receipt.json"
    status_path = evidence_dir / "quick_replay_status.json"
    receipt_path.unlink(missing_ok=True)
    run_started = utc_now()
    atomic_json(status_path, {"schema_version": 2, "status": "running", "started_at_utc": run_started})

    completed: list[dict[str, Any]] = []
    overall_started = time.perf_counter()
    try:
        config_resolved = config_path.resolve()
        try:
            config_resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Replay config must be inside the repository root") from exc
        if not config_resolved.is_file():
            raise FileNotFoundError(f"Replay config not found: {config_path}")
        config = yaml.safe_load(config_resolved.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Replay config must contain a YAML mapping")
        commands = config.get("commands")
        if not isinstance(commands, list) or len(commands) != 2:
            raise ValueError("Replay config must contain exactly two commands")
        validated = [validate_item(root, item, strand) for item, strand in zip(commands, STRANDS)]
        runner_record = relative_record(root, Path(__file__).resolve())
        replay_config_record = relative_record(root, config_resolved)

        for item in validated:
            remaining = MAX_TOTAL_SECONDS - (time.perf_counter() - overall_started)
            if remaining <= 0:
                raise TimeoutError("The complete two-strand replay exceeded ten minutes")
            before = output_snapshot(root, item["expected_output"])
            script_records = [relative_record(root, item["script"])]
            configuration_records = [relative_record(root, path) for path in item["configuration_inputs"]]
            data_records = [relative_record(root, path) for path in item["data_inputs"]]
            started_wall_ns = time.time_ns()
            started = time.perf_counter()
            subprocess.run(item["executed_argv"], cwd=root, check=True, timeout=remaining)
            after = output_snapshot(root, item["expected_output"])
            if not changed_since(before, after, started_wall_ns):
                raise RuntimeError(
                    f"{item['strand']} did not create or refresh its declared output; stale evidence is rejected"
                )
            require_unchanged(
                script_records,
                [relative_record(root, item["script"])],
                label=f"{item['strand']} script",
            )
            require_unchanged(
                configuration_records,
                [relative_record(root, path) for path in item["configuration_inputs"]],
                label=f"{item['strand']} configuration input",
            )
            require_unchanged(
                data_records,
                [relative_record(root, path) for path in item["data_inputs"]],
                label=f"{item['strand']} data input",
            )
            completed.append(
                {
                    "strand": item["strand"],
                    "declared_argv": item["declared_argv"],
                    "seconds": round(time.perf_counter() - started, 6),
                    "script": script_records[0],
                    "configuration_inputs": configuration_records,
                    "data_inputs": data_records,
                    "output_before": before,
                    "output_after": after,
                }
            )

        total_seconds = time.perf_counter() - overall_started
        if total_seconds > MAX_TOTAL_SECONDS:
            raise TimeoutError("The complete two-strand replay exceeded ten minutes")
        require_unchanged(
            [runner_record],
            [relative_record(root, Path(__file__).resolve())],
            label="replay runner",
        )
        require_unchanged(
            [replay_config_record],
            [relative_record(root, config_resolved)],
            label="replay configuration",
        )
        receipt = {
            "schema_version": 2,
            "status": "success",
            "started_at_utc": run_started,
            "completed_at_utc": utc_now(),
            "total_seconds": round(total_seconds, 6),
            "runner": runner_record,
            "replay_config": replay_config_record,
            "runs": completed,
        }
        atomic_json(receipt_path, receipt)
        atomic_json(
            status_path,
            {
                "schema_version": 2,
                "status": "success",
                "completed_at_utc": receipt["completed_at_utc"],
                "receipt": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": sha256(receipt_path),
            },
        )
        return receipt_path
    except Exception as exc:
        receipt_path.unlink(missing_ok=True)
        atomic_json(
            status_path,
            {
                "schema_version": 2,
                "status": "failed",
                "started_at_utc": run_started,
                "failed_at_utc": utc_now(),
                "completed_strands": [run["strand"] for run in completed],
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run_replays(args.config))


if __name__ == "__main__":
    main()
