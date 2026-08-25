#!/usr/bin/env python3
"""Create the manuscript task-triage comparison figure."""

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
TABLE = PROJECT / "artifacts" / "formal_validation" / "formal" / "tables" / "task_triage_for_manuscript.csv"
OUT = PROJECT / "artifacts" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE)
    task_labels = {
        "Bentall": "Bentall\nportable phenotype",
        "Arch_Fourbranch": "Four-branch arch\npractice shortcut",
        "Planned_CABG": "Planned CABG\nrare-task control",
    }
    series = [
        ("Random OOF\nphenotype", "Random_OOF_phenotype_AUROC", "#60A5FA"),
        ("Leave-one-surgeon-out\nphenotype", "Cross_surgeon_phenotype_AUROC", "#2563EB"),
        ("Random OOF\npractice only", "Random_OOF_practice_AUROC", "#F97316"),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    x = np.arange(len(data))
    width = 0.23
    for index, (label, column, color) in enumerate(series):
        position = x + (index - 1) * width
        bars = ax.bar(position, data[column], width=width, label=label, color=color, edgecolor="white")
        for bar, value in zip(bars, data[column]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#111827",
            )
    ax.axhline(0.5, color="#9CA3AF", linestyle="--", linewidth=1.2, label="Chance AUROC")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC")
    ax.set_xticks(x, [task_labels[item] for item in data["Task"]])
    ax.set_title("POP task triage: random accuracy, transportability, and practice shortcuts", pad=13)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"Figure2_task_triage.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
