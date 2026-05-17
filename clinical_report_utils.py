from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from sleep_aware_bp_framework import (
    first_wake_after_sleep,
    is_true,
    parse_measurement_datetime,
    summarise_participant,
)


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ClinicalThresholds:
    high_sbp_sd: float = 14.0
    high_morning_surge: float = 20.0


def load_reference_thresholds(root: Path = REPO_ROOT) -> ClinicalThresholds:
    threshold_path = root / "outputs" / "dryad_thresholds.csv"
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        main = thresholds.loc[thresholds["cohort"].eq("main")]
        if not main.empty:
            return ClinicalThresholds(
                high_sbp_sd=float(main.iloc[0]["sbp_sd_q75"]),
                high_morning_surge=float(main.iloc[0]["morning_surge_q75"]),
            )
    return ClinicalThresholds()


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("\xa0", "", regex=False).str.strip(),
        errors="coerce",
    )


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "id": "ID",
        "patient_id": "ID",
        "day_date": "Day_Date",
        "date": "Day_Date",
        "day date": "Day_Date",
        "time": "Time",
        "time_of_day": "Time",
        "systolic": "Systolic",
        "sys": "Systolic",
        "sbp": "Systolic",
        "bps": "Systolic",
        "diastolic": "Diastolic",
        "dia": "Diastolic",
        "dbp": "Diastolic",
        "bpd": "Diastolic",
        "map": "MAP",
        "pp": "PP",
        "pulse pressure": "PP",
        "hr": "HR",
        "pulse": "HR",
        "heart rate": "HR",
        "heart_rate": "HR",
        "wake_sleep": "Wake_Sleep",
        "wake/sleep": "Wake_Sleep",
        "sleep_wake": "Wake_Sleep",
        "awake_sleep": "Wake_Sleep",
        "state": "Wake_Sleep",
    }
    renamed = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace("-", "_")
        normalized = normalized.replace("__", "_")
        renamed[column] = aliases.get(normalized, aliases.get(normalized.replace("_", " "), column))
    return df.rename(columns=renamed)


def _parse_time_to_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return float(value.hour * 60 + value.minute + value.second / 60)
    parsed = pd.to_datetime(str(value).strip().split()[0], errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour * 60 + parsed.minute + parsed.second / 60)


def _derive_sleep_state(time_values: pd.Series, sleep_start: str, sleep_end: str) -> pd.Series:
    start = _parse_time_to_minutes(sleep_start)
    end = _parse_time_to_minutes(sleep_end)
    minutes = time_values.map(_parse_time_to_minutes)
    if np.isnan(start) or np.isnan(end):
        return pd.Series(1, index=time_values.index)
    if start <= end:
        asleep = minutes.ge(start) & minutes.lt(end)
    else:
        asleep = minutes.ge(start) | minutes.lt(end)
    return (~asleep).astype(int)


def _normalize_wake_sleep(series: pd.Series) -> pd.Series:
    mapped = []
    for value in series:
        if pd.isna(value):
            mapped.append(1)
            continue
        text = str(value).strip().lower()
        if text in {"0", "sleep", "asleep", "night", "n"}:
            mapped.append(0)
        elif text in {"1", "wake", "awake", "day", "w", "yes", "true"}:
            mapped.append(1)
        else:
            try:
                mapped.append(1 if float(text) > 0 else 0)
            except ValueError:
                mapped.append(1)
    return pd.Series(mapped, index=series.index, dtype=int)


def _fallback_datetimes(time_values: pd.Series) -> pd.Series:
    minutes = time_values.map(_parse_time_to_minutes)
    day_offsets = []
    offset = 0
    previous = None
    for value in minutes:
        if previous is not None and pd.notna(value) and pd.notna(previous) and value < previous:
            offset += 1
        day_offsets.append(offset)
        previous = value
    base = pd.Timestamp("2000-01-01")
    return pd.Series(
        [base + pd.Timedelta(days=day) + pd.Timedelta(minutes=value) for day, value in zip(day_offsets, minutes)],
        index=time_values.index,
    )


def prepare_patient_abpm(
    df: pd.DataFrame,
    patient_id: str = "NEW",
    sleep_start: str = "22:00",
    sleep_end: str = "07:00",
) -> pd.DataFrame:
    prepared = _canonical_columns(df.copy())
    required = {"Systolic", "Diastolic", "Time"}
    missing = sorted(required - set(prepared.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    prepared["ID_str"] = str(patient_id)
    for column in ["Systolic", "Diastolic"]:
        prepared[column] = _clean_numeric(prepared[column])
    if "MAP" not in prepared.columns:
        prepared["MAP"] = prepared["Diastolic"] + (prepared["Systolic"] - prepared["Diastolic"]) / 3
    else:
        prepared["MAP"] = _clean_numeric(prepared["MAP"])
    if "PP" not in prepared.columns:
        prepared["PP"] = prepared["Systolic"] - prepared["Diastolic"]
    else:
        prepared["PP"] = _clean_numeric(prepared["PP"])
    if "HR" not in prepared.columns:
        prepared["HR"] = np.nan
        hr_valid = pd.Series(True, index=prepared.index)
    else:
        prepared["HR"] = _clean_numeric(prepared["HR"])
        hr_valid = prepared["HR"].isna() | prepared["HR"].gt(0)

    if "Wake_Sleep" in prepared.columns:
        prepared["Wake_Sleep"] = _normalize_wake_sleep(prepared["Wake_Sleep"])
    else:
        prepared["Wake_Sleep"] = _derive_sleep_state(prepared["Time"], sleep_start, sleep_end)

    if "Day_Date" in prepared.columns:
        prepared["measurement_datetime"] = parse_measurement_datetime(prepared["Day_Date"], prepared["Time"])
    else:
        prepared["measurement_datetime"] = _fallback_datetimes(prepared["Time"])

    valid_mask = (
        prepared["Systolic"].gt(0)
        & prepared["Diastolic"].gt(0)
        & prepared["MAP"].gt(0)
        & hr_valid
        & prepared["measurement_datetime"].notna()
    )
    valid = prepared.loc[valid_mask].copy()
    if valid.empty:
        raise ValueError("No valid ABPM readings found after filtering.")

    valid = valid.sort_values("measurement_datetime").reset_index(drop=True)
    start = valid["measurement_datetime"].min()
    valid["hours_since_start"] = (valid["measurement_datetime"] - start).dt.total_seconds() / 3600
    return valid


def example_patient_abpm() -> pd.DataFrame:
    hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 23])
    sbp = np.array([139, 136, 138, 140, 137, 139, 136, 160, 164, 151, 136, 133, 140, 135, 142, 134, 139, 137])
    dbp = np.array([84, 82, 83, 85, 82, 84, 83, 96, 99, 91, 84, 80, 85, 82, 86, 81, 84, 83])
    base = pd.Timestamp("2026-01-01")
    df = pd.DataFrame(
        {
            "Time": [(base + pd.Timedelta(hours=float(h))).strftime("%H:%M:%S") for h in hours],
            "Systolic": sbp,
            "Diastolic": dbp,
            "HR": [72, 70, 69, 70, 71, 70, 72, 82, 86, 80, 75, 73, 74, 72, 76, 73, 72, 71],
        }
    )
    return prepare_patient_abpm(df, patient_id="EXAMPLE", sleep_start="22:00", sleep_end="07:00")


def build_patient_profile(valid: pd.DataFrame, thresholds: ClinicalThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or load_reference_thresholds()
    summary = summarise_participant(valid)
    summary["high_variability"] = bool(summary["sbp_sd"] >= thresholds.high_sbp_sd)
    summary["morning_surge_high"] = (
        bool(summary["morning_surge_sbp"] >= thresholds.high_morning_surge)
        if pd.notna(summary["morning_surge_sbp"])
        else False
    )
    summary["priority"] = monitoring_priority(summary)
    summary["pattern_flags"] = pattern_flags(summary)
    summary["review_points"] = review_points(summary)
    summary["patient_explanation"] = patient_explanation(summary)
    summary["data_quality"] = data_quality_label(summary)
    return summary


def data_quality_label(profile: dict[str, Any]) -> str:
    if profile["sleep_valid_readings"] < 3:
        return "Limited sleep BP data"
    if profile["valid_readings"] < 20:
        return "Limited valid readings"
    return "Adequate for profile review"


def monitoring_priority(profile: dict[str, Any]) -> str:
    if profile["dipping_category"] == "insufficient_sleep":
        return "Data review needed"
    if is_true(profile.get("sustained_high_bp")) or profile["dipping_category"] == "reverse_dipper":
        return "High review priority"
    if profile.get("morning_surge_high") and profile.get("high_variability"):
        return "High review priority"
    if (
        profile["dipping_category"] == "non_dipper"
        or profile.get("morning_surge_high")
        or profile.get("high_variability")
        or is_true(profile.get("hypertensive_sleep"))
    ):
        return "Review soon"
    return "Routine follow-up"


def pattern_flags(profile: dict[str, Any]) -> list[dict[str, str]]:
    dipping = profile.get("dipping_category")
    return [
        {"pattern": "Normal dipper", "status": "Yes" if dipping == "normal_dipper" else "No"},
        {"pattern": "Non-dipper", "status": "Yes" if dipping == "non_dipper" else "No"},
        {"pattern": "Reverse dipper", "status": "Yes" if dipping == "reverse_dipper" else "No"},
        {"pattern": "Morning surge", "status": "Yes" if profile.get("morning_surge_high") else "No"},
        {"pattern": "High variability", "status": "Yes" if profile.get("high_variability") else "No"},
        {"pattern": "Sustained high BP", "status": "Yes" if is_true(profile.get("sustained_high_bp")) else "No"},
    ]


def review_points(profile: dict[str, Any]) -> list[dict[str, str]]:
    points = []
    dipping = profile.get("dipping_category")
    if dipping == "insufficient_sleep":
        points.append({"Detected pattern": "Limited sleep data", "Doctor review point": "Review ABPM quality or consider repeat monitoring"})
    if dipping == "non_dipper":
        points.append({"Detected pattern": "Non-dipper", "Doctor review point": "Review night BP and sleep quality"})
    if dipping == "reverse_dipper":
        points.append({"Detected pattern": "Reverse dipper", "Doctor review point": "Prioritise nocturnal BP pattern review"})
    if profile.get("morning_surge_high"):
        points.append({"Detected pattern": "Morning surge", "Doctor review point": "Review morning BP control and medication timing with clinician"})
    if profile.get("high_variability"):
        points.append({"Detected pattern": "High variability", "Doctor review point": "Check stress, caffeine, pain, missed medication and measurement quality"})
    if is_true(profile.get("sustained_high_bp")):
        points.append({"Detected pattern": "Sustained high BP", "Doctor review point": "Consider earlier treatment review"})
    if not points:
        points.append({"Detected pattern": "No major rule-based flag", "Doctor review point": "Routine follow-up if clinically appropriate"})
    return points


def patient_explanation(profile: dict[str, Any]) -> str:
    parts = []
    if profile.get("dipping_category") in {"non_dipper", "reverse_dipper"}:
        parts.append("Your blood pressure did not fall enough during sleep.")
    if profile.get("morning_surge_high"):
        parts.append("Your blood pressure increased after waking.")
    if is_true(profile.get("hypertensive_sleep")):
        parts.append("Your blood pressure stayed high during sleep.")
    if profile.get("high_variability"):
        parts.append("Your readings changed more than expected over the day.")
    if not parts:
        parts.append("Your 24-hour blood pressure pattern did not show a major rule-based warning flag.")
    parts.append("Your doctor can use this report to review night-time BP control, sleep quality, stress, caffeine intake and medication timing.")
    return " ".join(parts)


def profile_label(profile: dict[str, Any]) -> str:
    labels = []
    dipping_map = {
        "normal_dipper": "Normal dipper",
        "non_dipper": "Non-dipper",
        "reverse_dipper": "Reverse dipper",
        "extreme_dipper": "Extreme dipper",
        "insufficient_sleep": "Insufficient sleep BP data",
    }
    labels.append(dipping_map.get(profile.get("dipping_category"), "Unclassified"))
    if profile.get("morning_surge_high"):
        labels.append("morning surge")
    if profile.get("high_variability"):
        labels.append("high variability")
    if is_true(profile.get("sustained_high_bp")):
        labels.append("sustained high BP")
    return " with ".join([labels[0], ", ".join(labels[1:])]) if len(labels) > 1 else labels[0]


def priority_level(priority: str) -> str:
    if priority == "High review priority":
        return "red"
    if priority in {"Review soon", "Data review needed"}:
        return "yellow"
    return "green"


def plot_24h_bp_curve(valid: pd.DataFrame, profile: dict[str, Any], patient_view: bool = False):
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = valid["hours_since_start"]

    for start, end in _sleep_segments(valid):
        ax.axvspan(start, end, color="#e8eef7", alpha=0.9)

    wake_time = first_wake_after_sleep(valid)
    if wake_time is not None:
        start = (wake_time - valid["measurement_datetime"].min()).total_seconds() / 3600
        ax.axvspan(start, start + 2, color="#f7dfb9", alpha=0.6)
        ax.axvline(start, color="#9a6a23", linestyle="--", linewidth=1.1)

    ax.plot(x, valid["Systolic"], color="#2f5d7c", marker="o", linewidth=2, label="Systolic BP")
    if not patient_view:
        ax.plot(x, valid["Diastolic"], color="#7b587b", marker="o", linewidth=1.8, label="Diastolic BP")
        ax.axhline(135, color="#9a6a23", linestyle="--", linewidth=1.1, label="SBP review line")
        ax.axhline(85, color="#7b587b", linestyle=":", linewidth=1.1, label="DBP review line")
    else:
        ax.axhline(135, color="#9a6a23", linestyle="--", linewidth=1.1)

    ax.set_title("24-hour BP curve" if not patient_view else "Your 24-hour blood pressure pattern", loc="left", fontweight="bold")
    ax.set_xlabel("Hours since first reading")
    ax.set_ylabel("BP (mmHg)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def _sleep_segments(valid: pd.DataFrame) -> list[tuple[float, float]]:
    ordered = valid.sort_values("hours_since_start").reset_index(drop=True)
    segments = []
    sleep_rows = ordered.loc[ordered["Wake_Sleep"].eq(0)]
    if sleep_rows.empty:
        return segments
    start = None
    last = None
    for _, row in ordered.iterrows():
        if row["Wake_Sleep"] == 0 and start is None:
            start = float(row["hours_since_start"])
        if row["Wake_Sleep"] != 0 and start is not None:
            segments.append((start, float(last if last is not None else row["hours_since_start"])))
            start = None
        last = row["hours_since_start"]
    if start is not None:
        segments.append((start, float(ordered["hours_since_start"].max())))
    return segments


def plot_profile_position(profile: dict[str, Any]):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axvspan(-5, 0, color="#f6d6d2", alpha=0.75)
    ax.axvspan(0, 10, color="#f7dfb9", alpha=0.75)
    ax.axvspan(10, 20, color="#dcefe4", alpha=0.85)
    ax.axvspan(20, 30, color="#dfe8f7", alpha=0.75)
    ax.axhspan(20, 35, color="#f5e3d2", alpha=0.45)
    ax.axvline(0, color="#8a3b35", linewidth=1)
    ax.axvline(10, color="#587a62", linewidth=1)
    ax.axvline(20, color="#587a62", linewidth=1)
    ax.axhline(20, color="#9a6a23", linestyle="--", linewidth=1.2)

    dipping = profile.get("dipping_pct_sbp")
    surge = profile.get("morning_surge_sbp")
    if pd.notna(dipping) and pd.notna(surge):
        ax.scatter([dipping], [surge], s=160, color="#b53d2f", edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate("Patient", xy=(dipping, surge), xytext=(dipping + 3, surge + 4), arrowprops=dict(arrowstyle="->", color="#334e5c"), fontsize=9)

    ax.text(-2.5, 5, "Reverse", ha="center", fontsize=8)
    ax.text(5, 5, "Non-dipper", ha="center", fontsize=8)
    ax.text(15, 5, "Normal\ndipper", ha="center", fontsize=8)
    ax.text(25, 5, "Extreme\ndipper", ha="center", fontsize=8)
    ax.text(21, 22.2, "High morning surge region", fontsize=8, color="#7a4e16")
    ax.set_xlim(-5, 30)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Sleep SBP dipping (%)")
    ax.set_ylabel("Morning surge (mmHg)")
    ax.set_title("Profile position plot", loc="left", fontweight="bold")
    ax.grid(alpha=0.15)
    return fig


def plot_awake_sleep_bar(profile: dict[str, Any]):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["Awake", "Sleep"]
    sbp = [profile.get("awake_mean_sbp"), profile.get("sleep_mean_sbp")]
    dbp = [profile.get("awake_mean_dbp"), profile.get("sleep_mean_dbp")]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, sbp, width=0.35, label="SBP", color="#2f5d7c")
    ax.bar(x + 0.18, dbp, width=0.35, label="DBP", color="#7b587b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BP (mmHg)")
    ax.set_title("Awake vs sleep BP", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    return fig


def feature_table(profile: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Measure": "Awake SBP", "Result": f"{profile['awake_mean_sbp']:.0f}", "Interpretation": "High" if profile["awake_mean_sbp"] >= 135 else "Within review range"},
            {"Measure": "Sleep SBP", "Result": f"{profile['sleep_mean_sbp']:.0f}" if pd.notna(profile["sleep_mean_sbp"]) else "N/A", "Interpretation": "High" if pd.notna(profile["sleep_mean_sbp"]) and profile["sleep_mean_sbp"] >= 120 else "Review with context"},
            {"Measure": "Dipping %", "Result": f"{profile['dipping_pct_sbp']:.1f}%" if pd.notna(profile["dipping_pct_sbp"]) else "N/A", "Interpretation": str(profile["dipping_category"]).replace("_", " ").title()},
            {"Measure": "Morning surge", "Result": f"{profile['morning_surge_sbp']:.0f} mmHg" if pd.notna(profile["morning_surge_sbp"]) else "N/A", "Interpretation": "High" if profile.get("morning_surge_high") else "Not flagged"},
            {"Measure": "SBP variability", "Result": f"{profile['sbp_sd']:.1f}", "Interpretation": "High" if profile.get("high_variability") else "Not flagged"},
            {"Measure": "Pulse pressure", "Result": f"{profile['mean_pp']:.0f} mmHg", "Interpretation": "Review if clinically relevant"},
        ]
    )


def create_pdf_report(
    valid: pd.DataFrame,
    profile: dict[str, Any],
    patient_details: dict[str, str],
    assistant_summary: str | None = None,
) -> bytes:
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _pdf_summary_page(pdf, profile, patient_details)
        _pdf_graph_page(pdf, valid, profile)
        _pdf_feature_table_page(pdf, profile)
        _pdf_checklist_page(pdf)
        if assistant_summary:
            _pdf_assistant_page(pdf, assistant_summary)
    buffer.seek(0)
    return buffer.read()


def _pdf_summary_page(pdf: PdfPages, profile: dict[str, Any], details: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    lines = [
        "Sleep-Aware BP Profile Report",
        "",
        f"Patient ID: {details.get('Patient ID', '')}",
        f"Age: {details.get('Age', '')}",
        f"Sex: {details.get('Sex', '')}",
        f"BMI: {details.get('BMI', '')}",
        f"ABPM date: {details.get('ABPM date', '')}",
        f"Valid readings: {profile['valid_readings']}",
        f"Sleep readings: {profile['sleep_valid_readings']}",
        f"Data quality: {profile['data_quality']}",
        "",
        f"Overall profile: {profile_label(profile)}",
        f"Monitoring priority: {profile['priority']}",
        "",
        "Main review point:",
        "Review night BP, morning BP control, sleep quality, adherence, caffeine/stress triggers and medication timing as clinically appropriate.",
    ]
    ax.text(0.08, 0.92, "\n".join(lines), va="top", fontsize=13, linespacing=1.45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_graph_page(pdf: PdfPages, valid: pd.DataFrame, profile: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    source_fig = plot_24h_bp_curve(valid, profile)
    _move_axes(source_fig.axes[0], fig, grid[0, :])
    plt.close(source_fig)
    source_fig = plot_awake_sleep_bar(profile)
    _move_axes(source_fig.axes[0], fig, grid[1, 0])
    plt.close(source_fig)
    source_fig = plot_profile_position(profile)
    _move_axes(source_fig.axes[0], fig, grid[1, 1])
    plt.close(source_fig)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _move_axes(old_ax, new_fig, subplot_spec):
    old_fig = old_ax.figure
    old_ax.remove()
    old_ax.figure = new_fig
    new_fig.axes.append(old_ax)
    new_fig.add_axes(old_ax)
    old_ax.set_subplotspec(subplot_spec)
    old_fig.clear()


def _pdf_feature_table_page(pdf: PdfPages, profile: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    table_df = feature_table(profile)
    ax.text(0.05, 0.95, "Feature table", fontsize=18, weight="bold", va="top")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.05, 0.35, 0.9, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_checklist_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    checklist = [
        "Doctor review checklist",
        "",
        "[ ] Review home BP technique",
        "[ ] Review adherence",
        "[ ] Review sleep quality",
        "[ ] Review caffeine/alcohol/stress triggers",
        "[ ] Review morning BP control",
        "[ ] Review medication timing if clinically appropriate",
        "[ ] Consider repeat ABPM or home BP follow-up",
        "",
        "Patient safety note:",
        "Continue prescribed medicine. Do not change dose without clinician advice. Seek urgent help if high BP is associated with chest pain, weakness, severe headache, confusion or shortness of breath.",
    ]
    ax.text(0.08, 0.92, "\n".join(checklist), va="top", fontsize=14, linespacing=1.7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_assistant_page(pdf: PdfPages, assistant_summary: str) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.08, 0.94, "Ask About This BP Report", fontsize=18, weight="bold", va="top")
    ax.text(
        0.08,
        0.88,
        "This page explains the already calculated report. It is not a diagnosis or medication instruction.",
        fontsize=10,
        va="top",
        color="#4b5563",
    )
    wrapped = "\n\n".join(
        "\n".join(textwrap.wrap(paragraph, width=82))
        for paragraph in str(assistant_summary).split("\n\n")
    )
    ax.text(0.08, 0.80, wrapped, va="top", fontsize=11.5, linespacing=1.45)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
