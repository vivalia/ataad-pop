#!/usr/bin/env python3
"""Create the manuscript POP framework and leakage-control schematic."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "artifacts" / "figures"


COLORS = {
    "phenotype": "#DBEAFE",
    "phenotype_edge": "#2563EB",
    "observation": "#FEF3C7",
    "observation_edge": "#D97706",
    "practice": "#FEE2E2",
    "practice_edge": "#DC2626",
    "neutral": "#F3F4F6",
    "neutral_edge": "#4B5563",
    "success": "#D1FAE5",
    "success_edge": "#059669",
    "ink": "#111827",
    "muted": "#6B7280",
}


def box(ax, x, y, w, h, text, face, edge, fontsize=10.5, linewidth=1.5, radius=0.02):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color="#4B5563", style="-|>", linewidth=1.5, connection="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=linewidth,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.2, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Phenotype–Observation–Practice (POP) transportability framework",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.925,
        "Signal provenance is separated before modeling; every preprocessing step is contained in the outer training environment",
        ha="center",
        va="top",
        fontsize=10.8,
        color=COLORS["muted"],
    )

    box(
        ax,
        0.055,
        0.78,
        0.25,
        0.105,
        "PHENOTYPE\nStrictly preoperative patient data\nclinical · laboratory · echo · CTA",
        COLORS["phenotype"],
        COLORS["phenotype_edge"],
        fontsize=10.5,
    )
    box(
        ax,
        0.375,
        0.78,
        0.25,
        0.105,
        "OBSERVATION\nWhich tests or values are missing\n27 eligible missingness indicators",
        COLORS["observation"],
        COLORS["observation_edge"],
        fontsize=10.5,
    )
    box(
        ax,
        0.695,
        0.78,
        0.25,
        0.105,
        "PRACTICE\nSurgeon and calendar context\nworkflow and procedure-preference signals",
        COLORS["practice"],
        COLORS["practice_edge"],
        fontsize=10.5,
    )

    ax.text(0.18, 0.747, "confirmatory prediction", ha="center", fontsize=9.2, color=COLORS["phenotype_edge"])
    ax.text(0.50, 0.747, "sensitivity + drift audit", ha="center", fontsize=9.2, color=COLORS["observation_edge"])
    ax.text(0.82, 0.747, "shortcut audit only", ha="center", fontsize=9.2, color=COLORS["practice_edge"])

    outer = FancyBboxPatch(
        (0.035, 0.26),
        0.665,
        0.45,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="#FFFFFF",
        edgecolor=COLORS["phenotype_edge"],
        linewidth=2.0,
    )
    ax.add_patch(outer)
    ax.text(
        0.055,
        0.682,
        "OUTER ENVIRONMENT VALIDATION",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["phenotype_edge"],
    )
    ax.text(
        0.685,
        0.682,
        "6 held-out surgeons  |  rolling test years 2020–2024",
        ha="right",
        fontsize=9.3,
        color=COLORS["muted"],
    )

    box(
        ax,
        0.07,
        0.565,
        0.17,
        0.085,
        "Outer TRAIN\nenvironments only",
        COLORS["phenotype"],
        COLORS["phenotype_edge"],
        fontsize=10.2,
    )
    box(
        ax,
        0.285,
        0.565,
        0.17,
        0.085,
        "Fold-specific filter\n>30% missing / constant",
        COLORS["neutral"],
        COLORS["neutral_edge"],
        fontsize=9.8,
    )
    box(
        ax,
        0.50,
        0.565,
        0.165,
        0.085,
        "20 stochastic MICE\nfit independently",
        COLORS["neutral"],
        COLORS["neutral_edge"],
        fontsize=9.8,
    )
    arrow(ax, (0.24, 0.607), (0.285, 0.607), color=COLORS["phenotype_edge"])
    arrow(ax, (0.455, 0.607), (0.50, 0.607), color=COLORS["phenotype_edge"])

    box(
        ax,
        0.07,
        0.425,
        0.17,
        0.085,
        "Outer TEST\nheld-out environment",
        "#FFFFFF",
        COLORS["phenotype_edge"],
        fontsize=10.2,
    )
    box(
        ax,
        0.285,
        0.425,
        0.17,
        0.085,
        "Transform only\nno refitting or tuning",
        COLORS["neutral"],
        COLORS["neutral_edge"],
        fontsize=9.8,
    )
    box(
        ax,
        0.50,
        0.425,
        0.165,
        0.085,
        "20 held-out\nprobability estimates",
        COLORS["neutral"],
        COLORS["neutral_edge"],
        fontsize=9.8,
    )
    arrow(ax, (0.24, 0.467), (0.285, 0.467), color=COLORS["phenotype_edge"])
    arrow(ax, (0.455, 0.467), (0.50, 0.467), color=COLORS["phenotype_edge"])
    arrow(ax, (0.582, 0.565), (0.582, 0.510), color=COLORS["phenotype_edge"], style="-[", linewidth=1.2)
    ax.text(0.594, 0.535, "training-fitted\nimputers", fontsize=8.2, color=COLORS["muted"], va="center")

    box(
        ax,
        0.13,
        0.305,
        0.225,
        0.075,
        "L2 logistic + inner 5-fold OOF\ntraining-only recalibration",
        COLORS["phenotype"],
        COLORS["phenotype_edge"],
        fontsize=9.7,
    )
    box(
        ax,
        0.425,
        0.305,
        0.225,
        0.075,
        "Pool MI predictions\nmean probability + between-MI SD",
        COLORS["success"],
        COLORS["success_edge"],
        fontsize=9.7,
    )
    arrow(ax, (0.37, 0.565), (0.29, 0.38), color=COLORS["phenotype_edge"], connection="arc3,rad=0.12")
    arrow(ax, (0.582, 0.425), (0.56, 0.38), color=COLORS["success_edge"])
    arrow(ax, (0.355, 0.342), (0.425, 0.342), color=COLORS["success_edge"])

    box(
        ax,
        0.735,
        0.575,
        0.225,
        0.105,
        "CHANNEL AUDIT\nRandom vs environment split\nphenotype vs observation vs practice",
        COLORS["neutral"],
        COLORS["neutral_edge"],
        fontsize=10.0,
    )
    box(
        ax,
        0.735,
        0.415,
        0.225,
        0.105,
        "TASK TRIAGE\nportable phenotype · shortcut case\nrare-task negative control",
        COLORS["neutral"],
        COLORS["neutral_edge"],
        fontsize=10.0,
    )
    arrow(ax, (0.50, 0.78), (0.80, 0.68), color=COLORS["observation_edge"], connection="arc3,rad=-0.18")
    arrow(ax, (0.82, 0.78), (0.86, 0.68), color=COLORS["practice_edge"])
    arrow(ax, (0.847, 0.575), (0.847, 0.52), color=COLORS["neutral_edge"])

    box(
        ax,
        0.055,
        0.09,
        0.245,
        0.105,
        "TRANSPORTABILITY\nAUROC · AUPRC · Brier · calibration\n2,000 environment-stratified bootstraps",
        COLORS["success"],
        COLORS["success_edge"],
        fontsize=9.7,
    )
    box(
        ax,
        0.375,
        0.09,
        0.245,
        0.105,
        "UNCERTAINTY-GUIDED DEFER\nselect within each surgeon environment\ncompare with random retention",
        COLORS["success"],
        COLORS["success_edge"],
        fontsize=9.7,
    )
    box(
        ax,
        0.695,
        0.09,
        0.25,
        0.105,
        "DEPLOYMENT STRESS\nworkflow missingness · no-echo boundary\nsubgroups · algorithm sensitivity",
        COLORS["success"],
        COLORS["success_edge"],
        fontsize=9.7,
    )
    arrow(ax, (0.48, 0.305), (0.22, 0.195), color=COLORS["success_edge"], connection="arc3,rad=0.08")
    arrow(ax, (0.54, 0.305), (0.50, 0.195), color=COLORS["success_edge"])
    arrow(ax, (0.60, 0.305), (0.78, 0.195), color=COLORS["success_edge"], connection="arc3,rad=-0.08")

    ax.text(
        0.5,
        0.033,
        "Intended output: human-supervised consistency review with explicit defer — not an autonomous treatment recommendation or causal benefit estimate",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.add_patch(Rectangle((0.02, 0.015), 0.96, 0.94, fill=False, edgecolor="#D1D5DB", linewidth=0.8))

    fig.tight_layout(pad=0.4)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"Figure1_POP_framework.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
