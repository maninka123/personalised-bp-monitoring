from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
OUTPUT_PATH = FIGURE_DIR / "new_patient_framework_example.png"


PATIENT = {
    "awake_mean_sbp": 140,
    "sleep_mean_sbp": 138,
    "dipping_pct": 1.4,
    "morning_surge": 24,
    "variability": "High",
    "profile": "Non-dipper with morning surge and high variability",
    "review": (
        "Review night BP, sleep quality, adherence, caffeine or stress triggers, "
        "and medication timing with clinician."
    ),
}


ML_VALIDATION = [
    {
        "target": "Circadian-Rythm",
        "feature": "Dipping / sleep fall",
        "metrics": "AUROC 0.991\nF1 0.955",
        "top": "Sys-Night-Des\nDia-Night-Des",
    },
    {
        "target": "Morning-Surge",
        "feature": "Morning rise",
        "metrics": "AUROC 0.941\nF1 0.697",
        "top": "BPS-wakeUp\nBPD-wakeUp",
    },
    {
        "target": "BP-Load",
        "feature": "24h BP burden",
        "metrics": "AUROC 0.980\nF1 0.986",
        "top": "BPS-load-Day\nBPS-Day24",
    },
    {
        "target": "Pulse-Pressure",
        "feature": "SBP-DBP gap",
        "metrics": "AUROC 0.732\nF1 0.828",
        "top": "Pressure-gap\nBP range",
    },
]


def add_card(ax, x, y, w, h, text, facecolor="#f8fafc", edgecolor="#425466", fontsize=9):
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(card)
    ax.text(
        x + 0.03,
        y + h - 0.05,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#1f2933",
        linespacing=1.25,
    )


def plot_bp_curve(ax):
    hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 23])
    sbp = np.array([139, 136, 138, 140, 137, 139, 136, 160, 164, 151, 136, 133, 140, 135, 142, 134, 139, 137])

    ax.axvspan(0, 7, color="#e8eef7", alpha=0.95, label="Sleep window")
    ax.axvspan(22, 24, color="#e8eef7", alpha=0.95)
    ax.axvspan(7, 9, color="#f7dfb9", alpha=0.75, label="First 2h after waking")
    ax.plot(hours, sbp, color="#2f5d7c", linewidth=2.2, marker="o", markersize=5)

    ax.axhline(135, color="#9a6a23", linestyle="--", linewidth=1.3, label="Awake SBP threshold 135")
    ax.axhline(120, color="#8c3f3b", linestyle=":", linewidth=1.5, label="Sleep SBP threshold 120")

    ax.set_xlim(0, 24)
    ax.set_ylim(110, 172)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Systolic BP (mmHg)")
    ax.set_title("A. New patient 24-hour BP curve", loc="left", fontweight="bold")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)

    ax.annotate(
        "Morning surge",
        xy=(8, 164),
        xytext=(10.4, 166),
        arrowprops=dict(arrowstyle="->", color="#7a4e16", linewidth=1.2),
        fontsize=9,
        color="#7a4e16",
    )
    ax.text(1.0, 114, "Sleep BP remains high", fontsize=8.5, color="#52616b")


def plot_profile_region(ax):
    ax.axvspan(-5, 0, color="#f6d6d2", alpha=0.75)
    ax.axvspan(0, 10, color="#f7dfb9", alpha=0.75)
    ax.axvspan(10, 20, color="#dcefe4", alpha=0.85)
    ax.axvspan(20, 30, color="#dfe8f7", alpha=0.75)
    ax.axhspan(20, 35, color="#f5e3d2", alpha=0.45)

    ax.axvline(0, color="#8a3b35", linewidth=1.1)
    ax.axvline(10, color="#587a62", linewidth=1.1)
    ax.axvline(20, color="#587a62", linewidth=1.1)
    ax.axhline(20, color="#9a6a23", linestyle="--", linewidth=1.4)

    ax.scatter(
        [PATIENT["dipping_pct"]],
        [PATIENT["morning_surge"]],
        s=220,
        marker="*",
        color="#b53d2f",
        edgecolor="white",
        linewidth=1.4,
        zorder=5,
    )
    ax.annotate(
        "Patient\n1.4% dip, 24 mmHg surge",
        xy=(PATIENT["dipping_pct"], PATIENT["morning_surge"]),
        xytext=(7.0, 29.0),
        arrowprops=dict(arrowstyle="->", color="#334e5c", linewidth=1.2),
        fontsize=8.7,
        color="#1f2933",
    )

    ax.text(-2.6, 6, "Reverse", ha="center", fontsize=8)
    ax.text(5, 6, "Non-dipper", ha="center", fontsize=8)
    ax.text(15, 6, "Normal\ndipper", ha="center", fontsize=8)
    ax.text(25, 6, "Extreme\ndipper", ha="center", fontsize=8)
    ax.text(22, 22.3, "High morning surge region", fontsize=8, color="#7a4e16")

    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Sleep SBP dipping (%)")
    ax.set_ylabel("Morning surge (mmHg)")
    ax.set_title("B. Profile region plot", loc="left", fontweight="bold")
    ax.grid(alpha=0.16)


def plot_report_card(ax):
    ax.axis("off")
    ax.set_title("C. Interpretable report", loc="left", fontweight="bold", pad=8)

    feature_text = "\n".join(
        [
            "Feature summary",
            "",
            f"Awake mean SBP:  {PATIENT['awake_mean_sbp']} mmHg",
            f"Sleep mean SBP:  {PATIENT['sleep_mean_sbp']} mmHg",
            f"Dipping:         {PATIENT['dipping_pct']}%",
            f"Morning surge:   {PATIENT['morning_surge']} mmHg",
            f"SBP variability: {PATIENT['variability']}",
        ]
    )
    add_card(ax, 0.02, 0.61, 0.96, 0.34, feature_text, "#eef4f8", fontsize=8.4)

    profile = textwrap.fill(PATIENT["profile"], width=38)
    profile_text = f"Assigned profile\n\n{profile}"
    add_card(ax, 0.02, 0.36, 0.96, 0.20, profile_text, "#f7f0df", fontsize=8.2)

    review = textwrap.fill(PATIENT["review"], width=46)
    review_text = f"Review point\n\n{review}"
    add_card(ax, 0.02, 0.04, 0.96, 0.26, review_text, "#f7e9e7", fontsize=7.8)


def plot_ml_validation(ax):
    ax.axis("off")
    ax.set_title("D. ML support validation", loc="left", fontweight="bold", pad=8)

    ml_text = (
        "The patient profile is assigned by transparent clinical rules.\n"
        "Kaggle ML checks whether similar ABPM feature groups predict related BP labels."
    )
    add_card(ax, 0.02, 0.82, 0.96, 0.15, ml_text, "#edf2f7", fontsize=7.8)

    table_rows = [
        [row["target"], row["feature"], row["metrics"], row["top"]]
        for row in ML_VALIDATION
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=["Target", "Related", "AUROC/F1", "Top signal"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.22, 0.96, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c9d3d9")
        if row == 0:
            cell.set_facecolor("#dcebf2")
            cell.set_text_props(weight="bold", color="#1f2933")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "#ffffff")

    note = (
        "ML does not make the final clinical decision. It supports the framework by "
        "showing that ABPM-derived feature groups can classify related abnormal BP patterns."
    )
    add_card(ax, 0.02, 0.02, 0.96, 0.14, textwrap.fill(note, width=58), "#eef4f8", fontsize=7.5)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#34495e",
            "axes.labelcolor": "#1f2933",
            "xtick.color": "#1f2933",
            "ytick.color": "#1f2933",
        }
    )

    fig = plt.figure(figsize=(15.2, 8.6), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 1.2, 1.18], height_ratios=[1, 1])

    ax_curve = fig.add_subplot(grid[0, 0:2])
    ax_region = fig.add_subplot(grid[1, 0:2])
    ax_report = fig.add_subplot(grid[0, 2])
    ax_ml = fig.add_subplot(grid[1, 2])

    plot_bp_curve(ax_curve)
    plot_profile_region(ax_region)
    plot_report_card(ax_report)
    plot_ml_validation(ax_ml)

    fig.suptitle(
        "How the Framework Handles a New Patient",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
