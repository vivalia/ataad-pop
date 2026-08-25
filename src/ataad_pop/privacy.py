"""Conservative checks that reduce accidental disclosure in a public repository."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PROHIBITED_SUFFIXES = {".xlsx", ".xls", ".parquet", ".feather", ".sav", ".dta"}
PROHIBITED_PATH_PARTS = {"predictions", "patient-level", "patient_level", "raw-data", "raw_data"}
DIRECT_IDENTIFIER_COLUMNS = {
    "name",
    "patient_name",
    "medical_record_number",
    "mrn",
    "national_id",
    "phone",
    "email",
    "address",
}
PATIENT_LEVEL_COLUMNS = {"row_id", "patient_id", "prediction", "predicted_probability"}


@dataclass(frozen=True)
class PrivacyFinding:
    path: Path
    reason: str


def _csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError):
        return set()
    return {item.strip().lower() for item in row}


def scan_public_tree(root: Path) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        relative = path.relative_to(root)
        lowered_parts = {part.lower() for part in relative.parts}
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            findings.append(PrivacyFinding(relative, "restricted tabular file type"))
        if lowered_parts.intersection(PROHIBITED_PATH_PARTS):
            findings.append(PrivacyFinding(relative, "patient-level output path"))
        if path.suffix.lower() == ".csv":
            header = _csv_header(path)
            identifiers = sorted(header.intersection(DIRECT_IDENTIFIER_COLUMNS))
            patient_outputs = sorted(header.intersection(PATIENT_LEVEL_COLUMNS))
            if identifiers:
                findings.append(PrivacyFinding(relative, f"direct identifiers: {identifiers}"))
            if len(patient_outputs) >= 2:
                findings.append(
                    PrivacyFinding(relative, f"probable patient-level table: {patient_outputs}")
                )
    return findings

