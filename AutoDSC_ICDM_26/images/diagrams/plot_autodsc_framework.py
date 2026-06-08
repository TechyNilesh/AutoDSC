#!/usr/bin/env python3
"""Render the AutoDSC framework diagram with embedded TrueType fonts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUT_DIR = Path(__file__).resolve().parent


def rounded_box(ax, x, y, width, height, facecolor, edgecolor, radius=0.12):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    return patch


def stage(ax, x, title, border, background, items):
    width, y, height = 3.25, 0.72, 3.05
    rounded_box(ax, x, y, width, height, background, border, radius=0.16)
    ax.text(
        x + width / 2,
        y + height - 0.25,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=border,
    )

    content_top = y + height - 0.62
    item_height = (height - 0.78) / len(items)
    for index, (heading, detail, bold) in enumerate(items):
        item_y = content_top - (index + 1) * item_height
        rounded_box(
            ax,
            x + 0.20,
            item_y + 0.07,
            width - 0.40,
            item_height - 0.10,
            "white",
            border,
            radius=0.05,
        )
        text_y = item_y + item_height / 2 + (0.08 if detail else 0)
        ax.text(
            x + width / 2,
            text_y,
            heading,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold" if bold else "normal",
        )
        if detail:
            ax.text(
                x + width / 2,
                text_y - 0.25,
                detail,
                ha="center",
                va="center",
                fontsize=8.2,
                color="#4A5568",
            )
    return x, y, width, height


def arrow(ax, start, end, label, y_offset=0.12, dashed=False, curved=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        linestyle="--" if dashed else "-",
        color="#4A5568",
        connectionstyle=f"arc3,rad={curved}",
    )
    ax.add_patch(patch)
    ax.text(
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2 + y_offset,
        label,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#4A5568",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )


def main():
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.35)
    ax.axis("off")
    ax.text(
        5.8,
        4.15,
        "AutoDSC Framework",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    stage(
        ax,
        0.15,
        "1) Data Stream Processing",
        "#1A5A9A",
        "#EBF8FF",
        [
            ("Data Stream", "", False),
            ("Configuration Population", "PredictOne + LearnOne", True),
            ("Real-Time Prediction", "from current incumbent", True),
        ],
    )
    stage(
        ax,
        4.18,
        "2) Evaluation and Adaptation",
        "#B54708",
        "#FFFAF0",
        [
            ("Normalised Composite Score", "SC | DBI | CHI", True),
            ("Incumbent Selection", "", False),
            ("Adaptive Exploration", "from score change", True),
            ("Diversity-Preserving Retention", "", True),
        ],
    )
    stage(
        ax,
        8.21,
        "3) PGSSO Search",
        "#237A4B",
        "#F0FFF4",
        [
            ("Recency-Weighted Elite Archive", "", True),
            ("Search Distributions", "algorithm frequencies + parameters", True),
            ("Challenger Generation", "", False),
        ],
    )

    arrow(ax, (3.40, 2.55), (4.18, 2.55), "assignments")
    arrow(ax, (7.43, 2.55), (8.21, 2.55), r"scores + $\alpha$")
    arrow(
        ax,
        (9.84, 0.72),
        (1.78, 0.72),
        "challengers",
        y_offset=-0.18,
        curved=-0.12,
    )
    arrow(
        ax,
        (4.18, 1.25),
        (3.40, 1.25),
        "retain",
        y_offset=-0.15,
        dashed=True,
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "autodsc_framework.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "autodsc_framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
