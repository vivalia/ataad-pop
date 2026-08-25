#!/usr/bin/env python3
"""Create manuscript Figure 5 and the supportive algorithm table."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", Path(__file__).resolve().parents[1]))
BENCHMARK = (
    PROJECT
    / "artifacts"
    / "formal_validation"
    / "formal"
    / "algorithm_benchmark"
    / "formal"
    / "tables"
)
FIGURES = PROJECT / "artifacts" / "figures"
TABLES = PROJECT / "artifacts" / "tables"

LABELS = {
    "L2_logistic_reproduction": "L2 logistic\n(confirmatory)",
    "Elastic_net_logistic": "Elastic-net logistic",
    "Random_forest": "Random forest",
    "LightGBM": "LightGBM",
}


def fmt_ci(row: pd.Series) -> str:
    return f"{row.Estimate:.3f} ({row.CI_lower:.3f}–{row.CI_upper:.3f})"


def make_table(metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for algorithm in LABELS:
        local = metrics.loc[metrics["Algorithm"].eq(algorithm)].set_index("Metric")
        delta = deltas.loc[
            deltas["Comparison_algorithm"].eq(algorithm)
            & deltas["Metric"].eq("AUROC")
        ]
        if algorithm == "L2_logistic_reproduction":
            auroc_delta = "Reference"
        else:
            d = delta.iloc[0]
            auroc_delta = (
                f"{d.Paired_delta_comparison_minus_reference:+.3f} "
                f"({d.Delta_CI_lower:+.3f} to {d.Delta_CI_upper:+.3f})"
            )
        auroc = local.loc["AUROC"]
        rows.append(
            {
                "Algorithm": LABELS[algorithm].replace("\n", " "),
                "AUROC_95CI": fmt_ci(auroc),
                "Paired_AUROC_delta_vs_L2_95CI": auroc_delta,
                "AUPRC_95CI": fmt_ci(local.loc["AUPRC"]),
                "Brier_95CI": fmt_ci(local.loc["Brier"]),
                "Calibration_intercept": auroc.Calibration_intercept,
                "Calibration_slope": auroc.Calibration_slope,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "Table3_supportive_algorithm_benchmark.csv", index=False)
    lines = [
        "| Algorithm | AUROC (95% CI) | Paired AUROC delta vs L2 (95% CI) | AUPRC (95% CI) | Brier (95% CI) | Calibration intercept | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.Algorithm} | {row.AUROC_95CI} | {row.Paired_AUROC_delta_vs_L2_95CI} | "
            f"{row.AUPRC_95CI} | {row.Brier_95CI} | {row.Calibration_intercept:.3f} | "
            f"{row.Calibration_slope:.3f} |"
        )
    (TABLES / "Table3_supportive_algorithm_benchmark.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_figure(metrics: pd.DataFrame) -> None:
    data = metrics.loc[metrics["Metric"].eq("AUROC")].copy()
    data["order"] = data["Algorithm"].map({name: i for i, name in enumerate(LABELS)})
    data = data.sort_values("order", ascending=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    y = np.arange(len(data))
    colors = ["#173F5F" if a == "L2_logistic_reproduction" else "#2F80A8" for a in data["Algorithm"]]
    for yi, (_, row), color in zip(y, data.iterrows(), colors):
        ax.errorbar(
            row["Estimate"],
            yi,
            xerr=np.array(
                [[row["Estimate"] - row["CI_lower"]], [row["CI_upper"] - row["Estimate"]]]
            ),
            fmt="none",
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            capthick=1.4,
            zorder=2,
        )
    ax.scatter(
        data["Estimate"],
        y,
        s=62,
        c=colors,
        marker="o",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for yi, row in zip(y, data.itertuples(index=False)):
        ax.text(
            row.CI_upper + 0.003,
            yi,
            f"{row.Estimate:.3f} ({row.CI_lower:.3f}–{row.CI_upper:.3f})",
            va="center",
            ha="left",
            fontsize=9,
            color="#20242A",
        )
    ax.set_yticks(y, [LABELS[a] for a in data["Algorithm"]])
    ax.set_xlim(0.755, 0.895)
    ax.set_xlabel("Leave-one-surgeon-out AUROC (95% CI)")
    ax.set_title("Fixed-hyperparameter supportive algorithm benchmark", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.0,
        -0.22,
        "All models: identical fold-specific features, 20 stochastic imputations, and training-only recalibration.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4C5560",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            FIGURES / f"Figure5_supportive_algorithm_benchmark.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(BENCHMARK / "algorithm_pooled_metrics.csv")
    deltas = pd.read_csv(BENCHMARK / "algorithm_paired_deltas_vs_l2.csv")
    make_table(metrics, deltas)
    make_figure(metrics)
    print("Created manuscript Figure 5 and Table 3.")


if __name__ == "__main__":
    main()
