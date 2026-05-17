from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"


def box(ax, xy, width, height, text, facecolor="#eef4f8", edgecolor="#48606f"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#1f2933",
        wrap=True,
    )


def arrow(ax, start, end, color="#334e5c"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color=color,
        )
    )


def save_dataset_usage():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "How the Two Datasets Are Used", ha="center", fontsize=16, weight="bold")

    box(ax, (0.06, 0.67), 0.23, 0.16, "Dryad\n24-hour ABPM + sleep/wake\nparticipant metadata", "#eaf4ef")
    box(ax, (0.38, 0.67), 0.23, 0.16, "Sleep-aware\nfeature extraction", "#eef4f8")
    box(ax, (0.70, 0.67), 0.23, 0.16, "Rule-based BP profiles\nand monitoring signals", "#f7f0df")

    box(ax, (0.06, 0.30), 0.23, 0.16, "Kaggle ABPM\nsummary features + labels", "#f1edf7")
    box(ax, (0.38, 0.30), 0.23, 0.16, "Machine learning\nclassification only", "#eef4f8")
    box(ax, (0.70, 0.30), 0.23, 0.16, "Metrics, confusion matrices\nand saved models", "#f7e9e7")

    arrow(ax, (0.29, 0.75), (0.38, 0.75))
    arrow(ax, (0.61, 0.75), (0.70, 0.75))
    arrow(ax, (0.29, 0.38), (0.38, 0.38))
    arrow(ax, (0.61, 0.38), (0.70, 0.38))

    ax.plot([0.05, 0.95], [0.57, 0.57], color="#c9d3d9", linewidth=1.2)
    ax.text(0.50, 0.535, "No participant-level row merge between Dryad and Kaggle", ha="center", fontsize=10, color="#52616b")
    ax.text(0.50, 0.17, "ML boundary: models are trained on Kaggle only", ha="center", fontsize=12, weight="bold", color="#8a3b35")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dataset_usage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_framework_outputs():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Analysis and Output Pipeline", ha="center", fontsize=16, weight="bold")

    box(ax, (0.04, 0.62), 0.18, 0.15, "Clean ABPM\nzero-row filtering", "#eaf4ef")
    box(ax, (0.29, 0.62), 0.18, 0.15, "Awake vs sleep\nBP features", "#eef4f8")
    box(ax, (0.54, 0.62), 0.18, 0.15, "BP profiles\nand risk flags", "#f7f0df")
    box(ax, (0.78, 0.62), 0.18, 0.15, "Clinical-review\nrecommendations", "#f7e9e7")

    arrow(ax, (0.22, 0.695), (0.29, 0.695))
    arrow(ax, (0.47, 0.695), (0.54, 0.695))
    arrow(ax, (0.72, 0.695), (0.78, 0.695))

    box(ax, (0.08, 0.25), 0.16, 0.14, "Dipping\nbar plot", "#f5f7fa")
    box(ax, (0.30, 0.25), 0.16, 0.14, "Awake vs sleep\nSBP plot", "#f5f7fa")
    box(ax, (0.52, 0.25), 0.16, 0.14, "Morning surge\ndistribution", "#f5f7fa")
    box(ax, (0.74, 0.25), 0.16, 0.14, "ML confusion\nmatrices", "#f5f7fa")

    arrow(ax, (0.38, 0.62), (0.16, 0.39), "#6b778d")
    arrow(ax, (0.38, 0.62), (0.38, 0.39), "#6b778d")
    arrow(ax, (0.63, 0.62), (0.60, 0.39), "#6b778d")
    arrow(ax, (0.63, 0.62), (0.82, 0.39), "#6b778d")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "framework_outputs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset_usage()
    save_framework_outputs()
    print(f"Saved README figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
