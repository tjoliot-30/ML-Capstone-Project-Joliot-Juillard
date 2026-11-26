"""Public paths, frozen-data loaders, and integrity helpers.

Synthetic target generation is intentionally absent from the student-facing
package. Frozen CSVs and their data cards are the public source of truth.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PUBLIC_SEEDS: tuple[int, ...] = (11, 23, 37, 51, 71)


def course_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_path(name: str) -> Path:
    return course_root() / "shared" / "datasets" / name


def load_frozen_csv(name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load one published CSV without synthesising or inferring hidden targets."""

    return pd.read_csv(dataset_path(name), **read_csv_kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

