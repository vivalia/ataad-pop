#!/usr/bin/env python3
"""Create manuscript-ready confirmatory and deployment performance tables."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", Path(__file__).resolve().parents[1]))
FORMAL = PROJECT / "artifacts" / "formal_validation" / "formal"
OUT = PROJECT / "artifacts" / "tables"


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def value_with_ci(frame: pd.DataFrame, metric: str) -> str:
    row = frame.loc[frame["Metric"].eq(metric)].iloc[0]
    return f"{fmt(row.Estimate)} ({fmt(row.CI_lower)}–{fmt(row.CI_upper)})"


def make_confirmatory_table() -> None:
    data = pd.read_csv(FORMAL / "tables" / "pooled_metrics_environment_bootstrap_ci.csv")
    rows: list[dict[str, object]] = []
    names = {
        "LEAVE_ONE_SURGEON_OUT": "Leave-one-surgeon-out",
        "ROLLING_YEAR": "Rolling year, 2020–2024",
    }
    models = {
        "Phenotype_only": "Phenotype only",
        "Phenotype_plus_observation": "Phenotype + observation indicators",
    }
    for scheme in names:
        for model in models:
            local = data.loc[data["Scheme"].eq(scheme) & data["Model"].eq(model)]
            first = local.iloc[0]
            rows.append({
                "Validation": names[scheme],
                "Model": models[model],
                "N": int(first.N),
                "Events": int(first.Events),
                "AUROC_95CI": value_with_ci(local, "AUROC"),
                "AUPRC_95CI": value_with_ci(local, "AUPRC"),
                "Brier_95CI": value_with_ci(local, "Brier"),
                "Calibration_intercept_95CI": value_with_ci(local, "Calibration_intercept"),
                "Calibration_slope_95CI": value_with_ci(local, "Calibration_slope"),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "Table2_confirmatory_performance.csv", index=False)
    lines = [
        "# Table II. Confirmatory transportability performance",
        "",
        "Values in parentheses are 95% environment-stratified bootstrap confidence intervals.",
        "",
        "| Validation | Model | N (events) | AUROC | AUPRC | Brier | Calibration intercept | Calibration slope |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.Validation} | {row.Model} | {row.N} ({row.Events}) | {row.AUROC_95CI} | "
            f"{row.AUPRC_95CI} | {row.Brier_95CI} | {row.Calibration_intercept_95CI} | "
            f"{row.Calibration_slope_95CI} |"
        )
    paired = pd.read_csv(FORMAL / "tables" / "paired_model_comparisons.csv")
    lines += ["", "Paired phenotype + observation minus phenotype-only differences:"]
    for scheme, label in names.items():
        auroc = paired.loc[paired["Scheme"].eq(scheme) & paired["Metric"].eq("AUROC")].iloc[0]
        brier = paired.loc[paired["Scheme"].eq(scheme) & paired["Metric"].eq("Brier")].iloc[0]
        lines.append(
            f"- {label}: AUROC Δ {fmt(auroc.Paired_delta_comparison_minus_reference)} "
            f"({fmt(auroc.Delta_CI_lower)} to {fmt(auroc.Delta_CI_upper)}); "
            f"Brier Δ {fmt(brier.Paired_delta_comparison_minus_reference)} "
            f"({fmt(brier.Delta_CI_lower)} to {fmt(brier.Delta_CI_upper)})."
        )
    (OUT / "Table2_confirmatory_performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_deployment_table() -> None:
    subgroup = pd.read_csv(FORMAL / "tables" / "supportive_subgroup_performance.csv")
    stress = pd.read_csv(FORMAL / "tables" / "missingness_shift_stress_summary.csv")
    selective = pd.read_csv(FORMAL / "tables" / "selective_performance.csv")
    rows: list[dict[str, object]] = []
    for row in subgroup.itertuples(index=False):
        label = "Overall" if row.Subgroup_variable == "Overall" else f"{row.Subgroup_variable}: {row.Level}"
        rows.append({
            "Analysis": "Subgroup",
            "Condition": label,
            "N": row.N,
            "Events": row.Events,
            "AUROC": row.AUROC,
            "AUROC_CI_lower": row.AUROC_CI_lower,
            "AUROC_CI_upper": row.AUROC_CI_upper,
            "AUPRC": row.AUPRC,
            "Brier": row.Brier,
            "Calibration_intercept": row.Calibration_intercept,
            "Calibration_slope": row.Calibration_slope,
        })
    for row in stress.itertuples(index=False):
        rows.append({
            "Analysis": "Missingness stress",
            "Condition": row.Scenario,
            "N": row.N,
            "Events": row.Events,
            "AUROC": row.AUROC,
            "AUROC_CI_lower": np.nan,
            "AUROC_CI_upper": np.nan,
            "AUPRC": row.AUPRC,
            "Brier": row.Brier,
            "Calibration_intercept": row.Calibration_intercept,
            "Calibration_slope": row.Calibration_slope,
        })
    for coverage in (1.0, 0.8, 0.6, 0.4):
        row = selective.loc[
            selective["Model"].eq("Phenotype_only")
            & selective["Selection_rule"].eq("Between_MI_SD")
            & selective["Coverage"].eq(coverage)
        ].iloc[0]
        rows.append({
            "Analysis": "Selective prediction",
            "Condition": f"Between-MI SD, {int(coverage * 100)}% coverage",
            "N": row.N,
            "Events": row.Events,
            "AUROC": row.AUROC,
            "AUROC_CI_lower": np.nan,
            "AUROC_CI_upper": np.nan,
            "AUPRC": row.AUPRC,
            "Brier": row.Brier,
            "Calibration_intercept": row.Calibration_intercept,
            "Calibration_slope": row.Calibration_slope,
        })
    pd.DataFrame(rows).to_csv(OUT / "TableS_deployment_stress_and_subgroups.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_confirmatory_table()
    make_deployment_table()
    print("Wrote manuscript confirmatory and deployment tables")


if __name__ == "__main__":
    main()
