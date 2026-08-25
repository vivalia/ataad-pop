#!/usr/bin/env python3
"""Create combined selective prediction and missingness stress figure."""

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
    selective = pd.read_csv(FORMAL / "tables" / "selective_performance.csv")
    selective = selective.loc[selective["Model"].eq("Phenotype_only")].copy()
    stress = pd.read_csv(FORMAL / "tables" / "missingness_shift_stress_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.25), gridspec_kw={"width_ratios": [1.05, 0.95]})
    ax = axes[0]
    styles = {
        "Between_MI_SD": ("Between-MI SD", "#2563EB", "o"),
        "Predictive_entropy": ("Predictive entropy", "#059669", "s"),
    }
    for rule, (label, color, marker) in styles.items():
        local = selective.loc[selective["Selection_rule"].eq(rule)].sort_values("Coverage")
        ax.plot(local["Coverage"] * 100, local["AUROC"], label=label, color=color, marker=marker, linewidth=2)
        for row in local.itertuples(index=False):
            ax.text(row.Coverage * 100, row.AUROC + 0.007, f"{row.AUROC:.3f}", ha="center", fontsize=8.3, color=color)
    random = selective.loc[selective["Selection_rule"].eq("Random_selection_reference")].sort_values("Coverage")
    x = random["Coverage"].to_numpy() * 100
    ax.fill_between(x, random["Random_AUROC_Q025"], random["Random_AUROC_Q975"], color="#D1D5DB", alpha=0.75, label="Random retention, 95% range")
    ax.plot(x, random["Random_AUROC_mean"], color="#6B7280", linestyle="--", linewidth=1.7, label="Random retention, mean")
    ax.set_xlim(35, 105)
    ax.set_ylim(0.76, 0.90)
    ax.set_xticks([40, 60, 80, 100])
    ax.set_xlabel("Coverage within each surgeon environment (%)")
    ax.set_ylabel("AUROC among retained patients")
    ax.set_title("A  MI-aware selective prediction")
    ax.grid(color="#E5E7EB", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.7, loc="upper right")

    ax = axes[1]
    plot = stress.loc[stress["Scenario"].ne("Observed_workflow")].copy()
    labels = {
        "Emulate_2023_2024_missingness": "2023–2024\nmissingness",
        "Worst_observed_year_missingness": "Worst observed\nyearly missingness",
        "No_preoperative_echo_available": "No preoperative\nechocardiography",
    }
    y = np.arange(len(plot))
    delta = plot["Delta_AUROC_vs_observed"].to_numpy()
    lower = plot["Delta_AUROC_CI_lower"].to_numpy()
    upper = plot["Delta_AUROC_CI_upper"].to_numpy()
    colors = ["#60A5FA", "#F59E0B", "#DC2626"]
    ax.barh(y, delta, color=colors, height=0.55)
    ax.errorbar(delta, y, xerr=np.vstack([delta - lower, upper - delta]), fmt="none", ecolor="#111827", capsize=3, linewidth=1.2)
    for position, value in zip(y, delta):
        ax.text(0.002, position, f"{value:.3f}", ha="left", va="center", fontsize=9, color="#111827", fontweight="bold")
    ax.axvline(0, color="#6B7280", linewidth=1)
    ax.set_yticks(y, [labels[item] for item in plot["Scenario"]])
    ax.set_xlim(-0.11, 0.012)
    ax.set_xlabel("Paired change in AUROC vs observed workflow")
    ax.set_title("B  Missingness-shift stress tests")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)

    for axis in axes:
        axis.set_axisbelow(True)
        sns.despine(ax=axis)
    fig.suptitle("Uncertainty can support deferral, but missing-domain failure remains explicit", fontsize=14, y=1.01)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"Figure4_selective_and_missingness_stress.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
