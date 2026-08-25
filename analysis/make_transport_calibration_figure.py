#!/usr/bin/env python3
"""Create pooled transportability forest and calibration panels."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", Path(__file__).resolve().parents[1]))
FORMAL = PROJECT / "artifacts" / "formal_validation" / "formal"
OUT = PROJECT / "artifacts" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(FORMAL / "tables" / "pooled_metrics_environment_bootstrap_ci.csv")
    calibration = pd.read_csv(FORMAL / "tables" / "calibration_bins.csv")
    auroc = metrics.loc[metrics["Metric"].eq("AUROC")].copy()
    order = [
        ("LEAVE_ONE_SURGEON_OUT", "Phenotype_only"),
        ("LEAVE_ONE_SURGEON_OUT", "Phenotype_plus_observation"),
        ("ROLLING_YEAR", "Phenotype_only"),
        ("ROLLING_YEAR", "Phenotype_plus_observation"),
    ]
    labels = {
        ("LEAVE_ONE_SURGEON_OUT", "Phenotype_only"): "Held-out surgeon\nPhenotype only",
        ("LEAVE_ONE_SURGEON_OUT", "Phenotype_plus_observation"): "Held-out surgeon\nPhenotype + observation",
        ("ROLLING_YEAR", "Phenotype_only"): "Rolling year\nPhenotype only",
        ("ROLLING_YEAR", "Phenotype_plus_observation"): "Rolling year\nPhenotype + observation",
    }
    rows = [
        auroc.loc[auroc["Scheme"].eq(scheme) & auroc["Model"].eq(model)].iloc[0]
        for scheme, model in order
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1.08, 0.92]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    colors = ["#2563EB", "#0F766E", "#2563EB", "#0F766E"]
    for position, row, color, key in zip(y, rows, colors, order):
        ax.errorbar(
            row.Estimate,
            position,
            xerr=[[row.Estimate - row.CI_lower], [row.CI_upper - row.Estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=7,
        )
        ax.text(row.CI_upper + 0.007, position, f"{row.Estimate:.3f}", va="center", fontsize=9.4, color=color)
    ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, [labels[key] for key in order])
    ax.set_xlim(0.48, 0.88)
    ax.set_xlabel("AUROC (95% environment-stratified bootstrap CI)")
    ax.set_title("A  Cross-environment discrimination")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)

    ax = axes[1]
    local = calibration.loc[calibration["Scheme"].eq("LEAVE_ONE_SURGEON_OUT")].copy()
    styles = {
        "Phenotype_only": ("Phenotype only", "#2563EB", "o"),
        "Phenotype_plus_observation": ("Phenotype + observation", "#0F766E", "s"),
    }
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.2, label="Ideal")
    for model, (label, color, marker) in styles.items():
        group = local.loc[local["Model"].eq(model)].sort_values("Predicted_mean")
        ax.plot(
            group["Predicted_mean"],
            group["Observed_rate"],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=label,
        )
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed Bentall proportion")
    ax.set_title("B  Pooled held-out-surgeon calibration")
    ax.grid(color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    for axis in axes:
        axis.set_axisbelow(True)
        sns.despine(ax=axis)
    fig.suptitle("Formal 20-imputation internal transportability validation", fontsize=14, y=1.01)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"Figure3_transportability_and_calibration.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
