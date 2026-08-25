#!/usr/bin/env python3
"""Create an observed-data cohort table for the JBHI manuscript."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", REPOSITORY))
RAW = Path(os.environ.get("ATAAD_POP_DATA", REPOSITORY / "data" / "TAAD_new1.xlsx"))
SHEET_NAME = os.environ.get("ATAAD_POP_SHEET", "main")
OUT = PROJECT / "artifacts" / "tables"

CONTINUOUS = [
    ("Age", "Age, years"),
    ("Onset_to_admission(d)", "Symptom onset to admission, days"),
    ("Preop_aortic_regurgitation", "Preoperative aortic-regurgitation measure"),
    ("LVEDD", "LV end-diastolic diameter"),
    ("LVEF", "Left-ventricular ejection fraction, %"),
    ("Hemoglobin", "Hemoglobin"),
    ("Creatinine", "Creatinine"),
    ("D_dimer", "D-dimer"),
]

BINARY = [
    ("Male", "Male sex"),
    ("Hypertension", "Hypertension"),
    ("CTD", "Connective tissue disorder"),
    ("Preop_hemodymical_unstable", "Preoperative hemodynamic instability"),
    ("First_tear_root", "First tear at aortic root"),
    ("First_tear_ascending_aorta", "First tear in ascending aorta"),
    ("Right_coronary_from_FL", "Right coronary artery from false lumen"),
    ("Cardiac_malperfusion", "Cardiac malperfusion"),
    ("Renal_malperfusion", "Renal malperfusion"),
    ("Pericardial_effusion", "Pericardial effusion"),
    ("Tamponade", "Tamponade"),
]


def fmt_continuous(series: pd.Series) -> str:
    observed = pd.to_numeric(series, errors="coerce").dropna()
    if observed.empty:
        return "NA"
    q1, median, q3 = observed.quantile([0.25, 0.5, 0.75])
    return f"{median:.1f} [{q1:.1f}–{q3:.1f}]"


def fmt_binary(series: pd.Series) -> str:
    observed = pd.to_numeric(series, errors="coerce").dropna()
    if observed.empty:
        return "NA"
    events = int(observed.eq(1).sum())
    return f"{events} ({100 * events / len(observed):.1f}%)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Conditional Formatting extension.*")
        if RAW.suffix.lower() == ".csv":
            data = pd.read_csv(RAW)
        else:
            data = pd.read_excel(RAW, sheet_name=SHEET_NAME)
    data.columns = data.columns.astype(str).str.strip()
    data["Row_ID"] = np.arange(1, len(data) + 1)
    data = data.loc[pd.to_numeric(data["Missingctimage"], errors="coerce").fillna(0).ne(1)].copy()
    mech = pd.to_numeric(data["Bentall_mechanic_valve"], errors="coerce").fillna(0)
    bio = pd.to_numeric(data["Bentall_bio_valve"], errors="coerce").fillna(0)
    data["Bentall"] = ((mech == 1) | (bio == 1)).astype(int)
    data["CTD"] = data["CTD"].replace({"1（类马方综合症）": 1})

    groups = [
        ("Overall", data),
        ("No Bentall", data.loc[data["Bentall"].eq(0)]),
        ("Bentall", data.loc[data["Bentall"].eq(1)]),
    ]
    rows: list[dict[str, object]] = []
    rows.append({
        "Characteristic": "Patients, N",
        "Type": "count",
        "Overall": str(len(data)),
        "No Bentall": str(int(data["Bentall"].eq(0).sum())),
        "Bentall": str(int(data["Bentall"].eq(1).sum())),
        "Overall_missing_n_pct": "0 (0.0%)",
    })
    for variable, label in CONTINUOUS:
        row: dict[str, object] = {"Characteristic": label, "Type": "median [IQR]"}
        for group_label, group in groups:
            row[group_label] = fmt_continuous(group[variable])
        missing = pd.to_numeric(data[variable], errors="coerce").isna().sum()
        row["Overall_missing_n_pct"] = f"{missing} ({100 * missing / len(data):.1f}%)"
        rows.append(row)
    for variable, label in BINARY:
        row = {"Characteristic": label, "Type": "n (% observed)"}
        for group_label, group in groups:
            row[group_label] = fmt_binary(group[variable])
        missing = pd.to_numeric(data[variable], errors="coerce").isna().sum()
        row["Overall_missing_n_pct"] = f"{missing} ({100 * missing / len(data):.1f}%)"
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "Table1_observed_cohort_characteristics.csv", index=False)

    lines = [
        "# Table I. Observed cohort characteristics by recorded Bentall decision",
        "",
        "Continuous variables are median [interquartile range]. Binary variables are n (% of observed values). No hypothesis tests were used for baseline description.",
        "",
        "| Characteristic | Overall | No Bentall | Bentall | Missing, n (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['Characteristic']} | {row['Overall']} | {row['No Bentall']} | "
            f"{row['Bentall']} | {row['Overall_missing_n_pct']} |"
        )
    (OUT / "Table1_observed_cohort_characteristics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote cohort table with {len(table)} rows")


if __name__ == "__main__":
    main()
