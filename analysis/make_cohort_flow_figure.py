#!/usr/bin/env python3
"""Create a TRIPOD-style cohort flow diagram for the ATAAD POP study."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ataad_jbhi_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


PROJECT = Path(os.environ.get("ATAAD_POP_OUTPUT_ROOT", Path(__file__).resolve().parents[1]))
OUT = PROJECT / "artifacts" / "figures"


def add_box(ax, xy, width, height, text, face, edge, fontsize=10, linestyle="-"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#18212B",
        linespacing=1.35,
    )
    return patch


def add_arrow(ax, start, end, color="#44515D"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Cohort assembly and analytic data availability",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#18212B",
    )

    add_box(
        ax,
        (0.16, 0.70),
        0.68,
        0.17,
        "Clinical source cohort\nN = 1,397\n\nAll available patients diagnosed with type A aortic dissection\nwho survived to operative treatment; no outcome- or procedure-based sampling",
        face="#EAF3F8",
        edge="#245B78",
        fontsize=9.7,
    )
    add_arrow(ax, (0.5, 0.695), (0.5, 0.615))

    add_box(
        ax,
        (0.18, 0.39),
        0.64,
        0.20,
        "Analytic data-availability exclusion\nN = 60\n\nResearch CT images or structured CT phenotype unavailable\n(Missingctimage = 1)\nNot an exclusion of the clinical ATAAD diagnosis",
        face="#FFF4E6",
        edge="#B56A1D",
        fontsize=9.5,
        linestyle="--",
    )
    add_arrow(ax, (0.5, 0.385), (0.5, 0.305))

    add_box(
        ax,
        (0.18, 0.11),
        0.64,
        0.17,
        "Confirmatory analytic cohort\nN = 1,337\n\nRecorded Bentall procedures: 389 (29.1%)\nSix surgeon environments; operations from 2016 to 2024",
        face="#EAF6EF",
        edge="#2C7550",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.035,
        "CT was the preoperative diagnostic reference; intraoperative findings were the diagnostic gold standard.",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#4A5560",
    )

    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUT / f"FigureS1_cohort_flow.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("Created Figure S1 cohort flow diagram.")


if __name__ == "__main__":
    main()
