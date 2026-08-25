"""Schema loading and validation for controlled or synthetic inputs."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = {
    "Date_of_surgery",
    "Surgeon",
    "Missingctimage",
    "Bentall_mechanic_valve",
    "Bentall_bio_valve",
}


def load_feature_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    required = {
        "Variable",
        "Domain",
        "Imputation_type",
        "Missing_indicator_for_sensitivity",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Feature metadata missing columns: {sorted(missing)}")
    if metadata["Variable"].duplicated().any():
        duplicates = metadata.loc[metadata["Variable"].duplicated(), "Variable"].tolist()
        raise ValueError(f"Duplicate feature metadata entries: {duplicates}")
    return metadata


def read_table(path: Path, sheet_name: str = "main") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Conditional Formatting extension.*")
            frame = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError("Input must be CSV, XLSX, or XLS")
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


def validate_input(frame: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    missing_core = sorted(CORE_COLUMNS.difference(frame.columns))
    if missing_core:
        issues.append(f"Missing core columns: {missing_core}")
    candidate_features = metadata["Variable"].astype(str).tolist()
    missing_features = sorted(set(candidate_features).difference(frame.columns))
    if missing_features:
        issues.append(f"Missing candidate features ({len(missing_features)}): {missing_features}")
    if "Surgeon" in frame:
        surgeon = pd.to_numeric(frame["Surgeon"], errors="coerce")
        if surgeon.notna().sum() == 0:
            issues.append("Surgeon contains no numeric values")
        elif surgeon.dropna().nunique() < 2:
            issues.append("At least two surgeon environments are required")
    if "Date_of_surgery" in frame:
        dates = pd.to_datetime(frame["Date_of_surgery"], errors="coerce")
        if dates.notna().sum() == 0:
            issues.append("Date_of_surgery contains no parseable dates")
    return issues


def derive_bentall(frame: pd.DataFrame) -> pd.Series:
    mech = pd.to_numeric(frame.get("Bentall_mechanic_valve"), errors="coerce").fillna(0)
    bio = pd.to_numeric(frame.get("Bentall_bio_valve"), errors="coerce").fillna(0)
    return ((mech == 1) | (bio == 1)).astype(np.int8)

