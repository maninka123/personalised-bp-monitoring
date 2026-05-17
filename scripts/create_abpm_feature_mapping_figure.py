from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
OUTPUT_PATH = FIGURE_DIR / "abpm_feature_mapping.png"


ROWS = [
    {
        "concept": "Overall BP level",
        "features": "BPS-24, BPD-24\nBPS-Day24, BPD-Day24\nBPS-Night24, BPD-Night24",
        "target": "BP-Load",
        "meaning": "High across\nboth day\nand night?",
    },
    {
        "concept": "Sleep BP fall",
        "features": "Sys-Night-Des\nDia-Night-Des\nNight BP features",
        "target": "Circadian-Rythm",
        "meaning": "Normal sleep\nBP fall?",
    },
    {
        "concept": "Morning rise",
        "features": "BPS-wakeUp\nBPD-wakeUp",
        "target": "Morning-Surge",
        "meaning": "Strong rise\nafter waking?",
    },
    {
        "concept": "BP spread and stiffness",
        "features": "Max-Sys, Min-Sys\nMax-Dia, Min-Dia\nPulse pressure label",
        "target": "Pulse-Pressure",
        "meaning": "Abnormal pressure\ngap or range?",
    },
    {
        "concept": "BP variability",
        "features": "BPS-CV-all/day/night\nBPD-CV-all/day/night",
        "target": "BP-Variability",
        "meaning": "Feature group only:\nnot a trained\ntarget here",
    },
]


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#425466", fontsize=9.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f2933",
        linespacing=1.18,
    )


def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.35,
            color="#52616b",
        )
    )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "How ABPM-Derived Features Map to the Kaggle ML Dataset",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#1f2933",
    )
    ax.text(
        0.5,
        0.915,
        "The Kaggle model uses ABPM summary features only. Dryad is used separately for raw sleep-aware BP profiling.",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#52616b",
    )

    columns = [
        (0.035, 0.835, 0.20, "Clinical ABPM idea"),
        (0.295, 0.835, 0.27, "Kaggle feature columns"),
        (0.615, 0.835, 0.15, "ML target label"),
        (0.835, 0.835, 0.145, "Beginner meaning"),
    ]
    for x, y, w, label in columns:
        draw_box(ax, x, y, w, 0.055, label, "#dcebf2", fontsize=10.5)

    y_positions = [0.70, 0.565, 0.43, 0.295, 0.16]
    colors = ["#eaf4ef", "#f7f0df", "#eef4f8", "#f7e9e7", "#f1edf7"]

    for row, y, color in zip(ROWS, y_positions, colors):
        draw_box(ax, 0.035, y, 0.20, 0.09, row["concept"], color)
        draw_box(ax, 0.295, y - 0.01, 0.27, 0.11, row["features"], "#f8fafc", fontsize=8.8)
        draw_box(ax, 0.615, y, 0.15, 0.09, row["target"], color, fontsize=9.4)
        draw_box(ax, 0.835, y - 0.005, 0.145, 0.10, row["meaning"], "#f8fafc", fontsize=7.6)
        draw_arrow(ax, (0.235, y + 0.045), (0.295, y + 0.045))
        draw_arrow(ax, (0.565, y + 0.045), (0.615, y + 0.045))
        draw_arrow(ax, (0.765, y + 0.045), (0.835, y + 0.045))

    draw_box(
        ax,
        0.17,
        0.04,
        0.66,
        0.06,
        "Interpretation: Kaggle tests whether ABPM summary features can classify patterns that Dryad calculates from raw 24-hour readings.",
        "#eef4f8",
        fontsize=8.8,
    )

    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
