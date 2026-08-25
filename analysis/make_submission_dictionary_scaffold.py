#!/usr/bin/env python3
"""Create the author-verification scaffold for supplementary Table S1."""

from __future__ import annotations

import csv
import os
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
INPUT = Path(
    os.environ.get("ATAAD_POP_FEATURE_METADATA", REPOSITORY / "config" / "feature_metadata.csv")
)
OUTPUT = PROJECT / "artifacts" / "tables" / "TableS1_candidate_variable_dictionary_scaffold.csv"


def main() -> None:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    rows = []
    for metadata in metadata_rows:
        rows.append(
            {
                "Variable": metadata["Variable"],
                "Domain": metadata["Domain"],
                "Imputation_type": metadata["Imputation_type"],
                "Missing_indicator_for_sensitivity": metadata.get(
                    "Missing_indicator_for_sensitivity", "False"
                ),
                "TRAIN_discrete_support": "",
            }
        )

    fieldnames = [
        "Variable",
        "Domain",
        "Imputation_type",
        "Missing_indicator_for_sensitivity",
        "TRAIN_discrete_support",
        "Clinical_display_name",
        "Unit",
        "Allowed_preoperative_measurement_window",
        "Multiple_measurement_selection_rule",
        "Source_system_or_acquisition_method",
        "Clinical_definition_or_coding_rule",
        "Plausibility_or_correction_rule",
        "Author_verification_status",
    ]

    for row in rows:
        row.update(
            {
                "Clinical_display_name": "",
                "Unit": "Not applicable" if row["Imputation_type"].startswith("discrete") else "",
                "Allowed_preoperative_measurement_window": "",
                "Multiple_measurement_selection_rule": "",
                "Source_system_or_acquisition_method": "",
                "Clinical_definition_or_coding_rule": "",
                "Plausibility_or_correction_rule": "",
                "Author_verification_status": "AUTHOR TO COMPLETE",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidate variables to {OUTPUT}")


if __name__ == "__main__":
    main()
