from __future__ import annotations

from pathlib import Path
import shutil
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"


def clean_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def true_count(series: pd.Series) -> int:
    return int(series.map(lambda value: False if pd.isna(value) else bool(value)).sum())


def save_table_csvs() -> None:
    features = pd.read_csv(OUTPUTS / "dryad_participant_features.csv")
    metrics = pd.read_csv(OUTPUTS / "kaggle_model_metrics.csv")
    labels = pd.read_csv(OUTPUTS / "kaggle_label_distribution.csv")
    coverage = pd.read_csv(OUTPUTS / "optional_physiology_coverage.csv")

    dryad_summary = pd.DataFrame(
        [
            {"Measure": "Raw sleep-annotated ABPM rows", "Result": "1,623"},
            {"Measure": "Valid rows after zero filtering", "Result": f"{len(pd.read_csv(OUTPUTS / 'dryad_valid_bp_readings.csv')):,}"},
            {"Measure": "Participants with valid ABPM", "Result": str(features["ID_str"].nunique())},
            {
                "Measure": "Insufficient sleep BP for dipping/surge",
                "Result": ", ".join(features.loc[features["dipping_category"].eq("insufficient_sleep"), "ID_str"].astype(str)),
            },
            {
                "Measure": "Normal dippers",
                "Result": str(int(features["dipping_category"].eq("normal_dipper").sum())),
            },
            {
                "Measure": "Non-dippers",
                "Result": str(int(features["dipping_category"].eq("non_dipper").sum())),
            },
            {
                "Measure": "Extreme dippers",
                "Result": str(int(features["dipping_category"].eq("extreme_dipper").sum())),
            },
            {
                "Measure": "High variability",
                "Result": str(int(features["high_variability"].sum())),
            },
            {
                "Measure": "High morning surge",
                "Result": str(int(features["morning_surge_high"].sum())),
            },
            {
                "Measure": "Sustained high BP",
                "Result": str(true_count(features["sustained_high_bp"])),
            },
        ]
    )
    dryad_summary.to_csv(TABLES / "table_1_dryad_profile_summary.csv", index=False)

    best_metrics = (
        metrics.sort_values(["target", "balanced_accuracy"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(1)
        .loc[:, ["target", "model", "auroc", "f1", "balanced_accuracy", "precision", "recall_sensitivity", "specificity"]]
    )
    best_metrics.to_csv(TABLES / "table_2_kaggle_best_models.csv", index=False)

    labels.to_csv(TABLES / "table_3_kaggle_label_distribution.csv", index=False)

    coverage_summary = pd.DataFrame(
        [
            {"Data stream": "ABPM exports", "Participants": int(coverage["has_abpm_export"].sum())},
            {"Data stream": "Zephyr summaries", "Participants": int(coverage["has_zephyr_summary"].sum())},
            {"Data stream": "CGM files", "Participants": int(coverage["has_cgm_file"].sum())},
            {"Data stream": "ECG segment files", "Participants": int(coverage["has_ecg_segments"].sum())},
            {"Data stream": "BP-merged ECG segments", "Participants": int(coverage["has_ecg_segments_bp_merged"].sum())},
        ]
    )
    coverage_summary.to_csv(TABLES / "table_4_optional_physiology_coverage.csv", index=False)


def copy_core_figures() -> None:
    source_figures = OUTPUTS / "figures"
    mapping = {
        "figure_1_framework_pipeline.png": "fig_1_framework_pipeline.png",
        "figure_2_dipping_categories.png": "fig_2_dipping_categories.png",
        "figure_3_awake_vs_sleep_sbp.png": "fig_3_awake_vs_sleep_sbp.png",
        "figure_4_morning_surge_distribution.png": "fig_4_morning_surge_distribution.png",
        "figure_5_example_bp_curves.png": "fig_5_example_bp_curves.png",
        "figure_6_kaggle_feature_importance.png": "fig_6_kaggle_feature_importance.png",
        "figure_7_clinical_monitoring_pathway.png": "fig_7_clinical_monitoring_pathway.png",
    }
    for src_name, dst_name in mapping.items():
        shutil.copy2(source_figures / src_name, FIGURES / dst_name)

    new_patient = ROOT / "docs" / "figures" / "new_patient_framework_example.png"
    if new_patient.exists():
        shutil.copy2(new_patient, FIGURES / "fig_8_new_patient_rule_based_report_with_ml_support.png")


def add_card(ax, x, y, w, h, text, face="#ffffff", edge="#cbd5e1", size=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0,
        facecolor=face,
        edgecolor=edge,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.03,
        y + h - 0.05,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        color="#0f172a",
        linespacing=1.28,
    )


def plot_dataset_complementarity() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")
    ax.set_title("How the Two Datasets Complement Each Other", loc="left", fontsize=15, fontweight="bold")

    dryad = (
        "Dryad physiological monitoring dataset\n\n"
        "Use: build the clinical framework\n"
        "- Raw 24-hour ABPM readings\n"
        "- Sleep/wake status\n"
        "- SBP, DBP, MAP, PP, HR\n"
        "- Participant factors and data notes\n\n"
        "Output: rule-based patient profile"
    )
    kaggle = (
        "Kaggle ABPM dataset\n\n"
        "Use: supporting ML validation\n"
        "- 270 labelled ABPM-summary records\n"
        "- 39 attributes\n"
        "- Labels for rhythm, pressure, load, surge\n\n"
        "Output: evidence that ABPM feature groups\n"
        "classify related BP pattern labels"
    )
    final = (
        "Combined interpretation\n\n"
        "Dryad answers:\n"
        "\"How does a patient's BP behave over 24 hours?\"\n\n"
        "Kaggle answers:\n"
        "\"Do similar ABPM features have predictive value\n"
        "for related abnormal BP labels?\"\n\n"
        "The cohorts are not merged row by row."
    )

    add_card(ax, 0.03, 0.25, 0.28, 0.62, dryad, "#eef7f2", "#7a9c83", 8.4)
    add_card(ax, 0.37, 0.25, 0.28, 0.62, kaggle, "#eef4fb", "#7d96b6", 8.4)
    add_card(ax, 0.70, 0.25, 0.27, 0.62, final, "#fff7ed", "#d6a05f", 8.2)

    ax.annotate("", xy=(0.36, 0.56), xytext=(0.32, 0.56), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))
    ax.annotate("", xy=(0.69, 0.56), xytext=(0.66, 0.56), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))

    ax.text(
        0.03,
        0.07,
        "Clinical boundary: the new-patient report is assigned by transparent rules. ML is a separate evidence panel, not a direct clinical decision engine.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_9_dataset_complementarity.png", dpi=240)
    plt.close(fig)


def plot_results_dashboard() -> None:
    features = pd.read_csv(OUTPUTS / "dryad_participant_features.csv")
    metrics = pd.read_csv(OUTPUTS / "kaggle_model_metrics.csv")
    labels = pd.read_csv(OUTPUTS / "kaggle_label_distribution.csv")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1])
    fig.suptitle("Main Results at a Glance", fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    counts = features["dipping_category"].value_counts().reindex(
        ["normal_dipper", "non_dipper", "extreme_dipper", "insufficient_sleep"], fill_value=0
    )
    ax1.bar([clean_label(x) for x in counts.index], counts.values, color=["#4f8a6a", "#c49a42", "#6277b8", "#8f8f8f"])
    ax1.set_title("Dryad rule-based profiles")
    ax1.set_ylabel("Participants")
    ax1.tick_params(axis="x", rotation=20)
    for i, value in enumerate(counts.values):
        ax1.text(i, value + 0.25, str(value), ha="center", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    flag_counts = pd.Series(
        {
            "High variability": int(features["high_variability"].sum()),
            "High morning surge": int(features["morning_surge_high"].sum()),
            "Sustained high BP": true_count(features["sustained_high_bp"]),
        }
    )
    ax2.bar(flag_counts.index, flag_counts.values, color=["#8b6f47", "#b56b45", "#a14d4d"])
    ax2.set_title("Clinician-review flags")
    ax2.set_ylabel("Participants")
    ax2.tick_params(axis="x", rotation=20)
    for i, value in enumerate(flag_counts.values):
        ax2.text(i, value + 0.15, str(value), ha="center", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    best = metrics.sort_values(["target", "balanced_accuracy"], ascending=[True, False]).groupby("target").head(1)
    best = best.sort_values("balanced_accuracy")
    y = np.arange(len(best))
    ax3.barh(y - 0.18, best["auroc"], height=0.35, label="AUROC", color="#4f6f7f")
    ax3.barh(y + 0.18, best["balanced_accuracy"], height=0.35, label="Balanced accuracy", color="#9a7b4f")
    ax3.set_yticks(y)
    ax3.set_yticklabels(best["target"])
    ax3.set_xlim(0, 1.05)
    ax3.set_title("Kaggle best model performance")
    ax3.legend(frameon=False, loc="lower right")

    ax4 = fig.add_subplot(gs[1, 1])
    target_labels = labels.loc[labels["label"].isin(["Circadian-Rythm", "Pulse-Pressure", "BP-Load", "Morning-Surge"])].copy()
    x = np.arange(len(target_labels))
    ax4.bar(x, target_labels["positive"], label="Positive", color="#5c7c8a")
    ax4.bar(x, target_labels["negative"], bottom=target_labels["positive"], label="Negative", color="#d9b77f")
    ax4.set_xticks(x)
    ax4.set_xticklabels(target_labels["label"], rotation=20)
    ax4.set_ylabel("Rows")
    ax4.set_title("Kaggle target label balance")
    ax4.legend(frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "fig_10_results_dashboard.png", dpi=240)
    plt.close(fig)


def write_caption_file() -> None:
    captions = {
        "fig_1_framework_pipeline.png": "Overall sleep-aware BP profiling pipeline from ABPM upload to clinician-review output.",
        "fig_2_dipping_categories.png": "Dryad participant distribution across rule-based dipping categories.",
        "fig_3_awake_vs_sleep_sbp.png": "Participant-level awake and sleep SBP relationship, with ABPM review thresholds.",
        "fig_4_morning_surge_distribution.png": "Distribution of Dryad morning surge values; high surge was cohort-relative top quartile.",
        "fig_5_example_bp_curves.png": "Example SBP curves showing different sleep-aware BP profiles.",
        "fig_6_kaggle_feature_importance.png": "Mean random forest feature importance across Kaggle ABPM classification targets.",
        "fig_7_clinical_monitoring_pathway.png": "Rule-based detected profile and safe monitoring recommendation pathway.",
        "fig_8_new_patient_rule_based_report_with_ml_support.png": "New-patient report workflow with separate Kaggle ML support validation panel.",
        "fig_9_dataset_complementarity.png": "Why Dryad and Kaggle are complementary rather than row-merged datasets.",
        "fig_10_results_dashboard.png": "Main Dryad and Kaggle results summarized in one figure.",
    }
    text = "\n".join(f"- **{name}:** {caption}" for name, caption in captions.items())
    (PAPER / "figure_captions.md").write_text("# Figure Captions\n\n" + text + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    save_table_csvs()
    copy_core_figures()
    plot_dataset_complementarity()
    plot_results_dashboard()
    write_caption_file()
    print(f"Paper assets written to {PAPER}")


if __name__ == "__main__":
    main()
