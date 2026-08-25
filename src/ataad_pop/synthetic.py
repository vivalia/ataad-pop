"""Generate a schema-compatible dataset that contains no real patient information."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import load_feature_metadata


def _continuous_feature(name: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if name == "Age":
        return np.clip(rng.normal(52, 11, n), 18, 85)
    if name == "LVEF":
        return np.clip(rng.normal(63, 6, n), 25, 80)
    if name in {"LVEDD", "LVESD"}:
        return np.clip(rng.normal(48 if name == "LVEDD" else 32, 7, n), 15, 90)
    if "diameter" in name.lower() or "width" in name.lower():
        return np.clip(rng.normal(35, 9, n), 5, 90)
    return np.clip(rng.lognormal(mean=1.6, sigma=0.55, size=n), 0, None)


def generate_synthetic(metadata_path: Path, n: int = 360, seed: int = 20260814) -> pd.DataFrame:
    if n < 60:
        raise ValueError("Synthetic examples require at least 60 rows")
    metadata = load_feature_metadata(metadata_path)
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2016-01-01") + pd.to_timedelta(
        rng.integers(0, 9 * 365, n), unit="D"
    )
    columns: dict[str, np.ndarray | pd.DatetimeIndex] = {
        "Date_of_surgery": dates,
        "Surgeon": np.tile(np.arange(1, 7), int(np.ceil(n / 6)))[:n],
        "Missingctimage": np.zeros(n, dtype=int),
    }

    for row in metadata.itertuples(index=False):
        kind = str(row.Imputation_type).lower()
        if "discrete" in kind:
            values = rng.binomial(1, 0.18, n).astype(float)
        else:
            values = _continuous_feature(str(row.Variable), rng, n).astype(float)
        if str(row.Missing_indicator_for_sensitivity).lower() == "true":
            values[rng.random(n) < 0.10] = np.nan
        columns[str(row.Variable)] = values

    frame = pd.DataFrame(columns)

    age = pd.to_numeric(frame.get("Age", 52), errors="coerce").fillna(52)
    ar = pd.to_numeric(frame.get("Preop_aortic_regurgitation", 0), errors="coerce").fillna(0)
    tear_root = pd.to_numeric(frame.get("First_tear_root", 0), errors="coerce").fillna(0)
    ctd = pd.to_numeric(frame.get("CTD", 0), errors="coerce").fillna(0)
    linear = -1.4 - 0.015 * (age - 52) + 0.20 * ar + 1.0 * tear_root + 0.8 * ctd
    probability = 1 / (1 + np.exp(-np.clip(linear, -8, 8)))
    bentall = rng.binomial(1, probability)
    mechanical = bentall * rng.binomial(1, 0.65, n)
    biological = bentall - mechanical
    outcome_frame = pd.DataFrame(
        {
            "Bentall_mechanic_valve": mechanical,
            "Bentall_bio_valve": biological,
        },
        index=frame.index,
    )
    return pd.concat([frame, outcome_frame], axis=1)


def write_synthetic(path: Path, metadata_path: Path, n: int, seed: int) -> Path:
    frame = generate_synthetic(metadata_path, n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".xlsx":
        frame.to_excel(path, sheet_name="main", index=False)
    else:
        raise ValueError("Synthetic output must end in .csv or .xlsx")
    return path
