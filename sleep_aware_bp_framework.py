from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DRYAD_DIR = ROOT / "24-hour physiological monitoring"
KAGGLE_ARFF = ROOT / "Kaggle dataset" / "y4dh3b3tfx-1" / "ABPM-dataset.arff"
OUTPUT_DIR = ROOT / "outputs"

MIN_SLEEP_READINGS = 3
RANDOM_STATE = 42

BP_COLUMNS = ["Systolic", "Diastolic", "MAP", "PP", "HR"]
VALIDITY_COLUMNS = ["Systolic", "Diastolic", "MAP", "HR"]
KAGGLE_LABELS = [
    "Validity",
    "Circadian-Rythm",
    "Pulse-Pressure",
    "BP-Variability",
    "BP-Load",
    "Morning-Surge",
]
KAGGLE_TARGETS = ["Circadian-Rythm", "Pulse-Pressure", "BP-Load", "Morning-Surge"]

AGE_GROUPS = {
    0: "19 and under",
    1: "20-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65 and over",
    11: "Prefer not to say",
}
SEX_LABELS = {0: "Male", 1: "Female"}


@dataclass(frozen=True)
class DryadThresholds:
    sbp_sd_q75: float
    morning_surge_q75: float


def format_participant_id(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{int(float(value)):03d}"
    except (TypeError, ValueError):
        return str(value).strip().zfill(3)


def time_to_string(value: object) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    return str(value).strip().split()[0]


def parse_measurement_datetime(day_date: pd.Series, time_of_day: pd.Series) -> pd.Series:
    dates = pd.to_datetime(day_date, dayfirst=True, errors="coerce")
    times = pd.to_timedelta(time_of_day.map(time_to_string), errors="coerce")
    return dates + times


def filter_valid_bp_readings(bp: pd.DataFrame) -> pd.DataFrame:
    filtered = bp.copy()
    for column in BP_COLUMNS:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
    valid_mask = filtered[VALIDITY_COLUMNS].gt(0).all(axis=1)
    return filtered.loc[valid_mask].copy()


def classify_dipping(
    awake_mean_sbp: float,
    sleep_mean_sbp: float,
    sleep_readings: int,
    min_sleep_readings: int = MIN_SLEEP_READINGS,
) -> tuple[float, str]:
    if (
        sleep_readings < min_sleep_readings
        or pd.isna(awake_mean_sbp)
        or pd.isna(sleep_mean_sbp)
        or awake_mean_sbp <= 0
    ):
        return np.nan, "insufficient_sleep"

    dipping_pct = ((awake_mean_sbp - sleep_mean_sbp) / awake_mean_sbp) * 100
    if dipping_pct < 0:
        category = "reverse_dipper"
    elif dipping_pct < 10:
        category = "non_dipper"
    elif dipping_pct <= 20:
        category = "normal_dipper"
    else:
        category = "extreme_dipper"
    return float(dipping_pct), category


def is_true(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


def load_dryad_bp(dryad_dir: Path = DRYAD_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(dryad_dir / "Blood_Pressure_Sleep_Info.xlsx")
    raw["ID_str"] = raw["ID"].map(format_participant_id)
    raw["measurement_datetime"] = parse_measurement_datetime(raw["Day_Date"], raw["Time"])

    valid = filter_valid_bp_readings(raw)
    valid = valid.sort_values(["ID_str", "measurement_datetime"]).reset_index(drop=True)
    valid["hours_since_start"] = valid.groupby("ID_str")["measurement_datetime"].transform(
        lambda s: (s - s.min()).dt.total_seconds() / 3600
    )
    return raw, valid


def load_participant_metadata(dryad_dir: Path = DRYAD_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(dryad_dir / "Participant_Information.csv")
    meta["ID_str"] = meta["ID"].map(format_participant_id)
    meta["sex_label"] = meta["Sex"].map(SEX_LABELS)
    meta["age_group_label"] = meta["Age"].map(AGE_GROUPS)

    notes = pd.read_csv(dryad_dir / "Data_Collection_Notes.csv")
    notes["ID_str"] = notes["ID"].map(format_participant_id)
    issue_text = notes["Issue reported/known?"].fillna("").astype(str).str.strip()
    notes["known_device_issue"] = ~issue_text.str.upper().isin(["", "N", "NO"])
    return meta, notes


def first_wake_after_sleep(group: pd.DataFrame) -> pd.Timestamp | None:
    ordered = group.sort_values("measurement_datetime").reset_index(drop=True)
    states = ordered["Wake_Sleep"].astype(int).to_numpy()
    for idx in range(1, len(ordered)):
        if states[idx] == 1 and states[idx - 1] == 0:
            return ordered.loc[idx, "measurement_datetime"]
    sleep_rows = ordered.loc[ordered["Wake_Sleep"].eq(0)]
    if sleep_rows.empty:
        return None
    last_sleep = sleep_rows["measurement_datetime"].max()
    awake_after_sleep = ordered.loc[
        ordered["Wake_Sleep"].eq(1) & ordered["measurement_datetime"].gt(last_sleep)
    ]
    if awake_after_sleep.empty:
        return None
    return awake_after_sleep["measurement_datetime"].min()


def summarise_participant(group: pd.DataFrame) -> dict[str, object]:
    group = group.sort_values("measurement_datetime")
    awake = group.loc[group["Wake_Sleep"].eq(1)]
    sleep = group.loc[group["Wake_Sleep"].eq(0)]

    awake_mean_sbp = awake["Systolic"].mean()
    awake_mean_dbp = awake["Diastolic"].mean()
    sleep_mean_sbp = sleep["Systolic"].mean()
    sleep_mean_dbp = sleep["Diastolic"].mean()
    dipping_pct, dipping_category = classify_dipping(awake_mean_sbp, sleep_mean_sbp, len(sleep))

    morning_surge_sbp = np.nan
    morning_window_readings = 0
    morning_surge_status = "insufficient_sleep"
    if len(sleep) >= MIN_SLEEP_READINGS:
        wake_time = first_wake_after_sleep(group)
        if wake_time is None:
            morning_surge_status = "no_wake_transition"
        else:
            window_end = wake_time + pd.Timedelta(hours=2)
            morning_window = group.loc[
                group["Wake_Sleep"].eq(1)
                & group["measurement_datetime"].ge(wake_time)
                & group["measurement_datetime"].lt(window_end)
            ]
            morning_window_readings = len(morning_window)
            if morning_window.empty:
                morning_surge_status = "no_morning_readings"
            else:
                morning_surge_sbp = morning_window["Systolic"].mean() - sleep_mean_sbp
                morning_surge_status = "available"

    sbp_sd = group["Systolic"].std(ddof=1)
    dbp_sd = group["Diastolic"].std(ddof=1)
    mean_sbp = group["Systolic"].mean()
    mean_dbp = group["Diastolic"].mean()

    if group["HR"].nunique() > 1 and group["Systolic"].nunique() > 1:
        hr_sbp_corr = group["HR"].corr(group["Systolic"])
    else:
        hr_sbp_corr = np.nan

    has_enough_sleep = len(sleep) >= MIN_SLEEP_READINGS
    hypertensive_24h = (mean_sbp >= 130) or (mean_dbp >= 80)
    hypertensive_awake = (awake_mean_sbp >= 135) or (awake_mean_dbp >= 85)
    hypertensive_sleep = (
        ((sleep_mean_sbp >= 120) or (sleep_mean_dbp >= 70)) if has_enough_sleep else pd.NA
    )
    sustained_high_bp = (
        bool(hypertensive_24h and hypertensive_awake and hypertensive_sleep)
        if has_enough_sleep
        else pd.NA
    )

    return {
        "ID_str": group["ID_str"].iloc[0],
        "valid_readings": len(group),
        "awake_valid_readings": len(awake),
        "sleep_valid_readings": len(sleep),
        "first_measurement": group["measurement_datetime"].min(),
        "last_measurement": group["measurement_datetime"].max(),
        "monitoring_hours": (
            group["measurement_datetime"].max() - group["measurement_datetime"].min()
        ).total_seconds()
        / 3600,
        "mean_24h_sbp": mean_sbp,
        "mean_24h_dbp": mean_dbp,
        "awake_mean_sbp": awake_mean_sbp,
        "awake_mean_dbp": awake_mean_dbp,
        "sleep_mean_sbp": sleep_mean_sbp,
        "sleep_mean_dbp": sleep_mean_dbp,
        "dipping_pct_sbp": dipping_pct,
        "dipping_category": dipping_category,
        "morning_surge_sbp": morning_surge_sbp,
        "morning_surge_status": morning_surge_status,
        "morning_window_readings": morning_window_readings,
        "sbp_sd": sbp_sd,
        "dbp_sd": dbp_sd,
        "sbp_cv_pct": (sbp_sd / mean_sbp) * 100 if mean_sbp else np.nan,
        "dbp_cv_pct": (dbp_sd / mean_dbp) * 100 if mean_dbp else np.nan,
        "mean_pp": group["PP"].mean(),
        "mean_map": group["MAP"].mean(),
        "mean_hr": group["HR"].mean(),
        "hr_sbp_corr": hr_sbp_corr,
        "hypertensive_24h": bool(hypertensive_24h),
        "hypertensive_awake": bool(hypertensive_awake),
        "hypertensive_sleep": hypertensive_sleep,
        "sustained_high_bp": sustained_high_bp,
    }


def monitoring_recommendation(row: pd.Series) -> str:
    if is_true(row.get("sustained_high_bp")):
        return "Early clinician review for sustained elevated ambulatory BP"
    if row.get("dipping_category") == "reverse_dipper":
        return "Prioritise clinician review of nocturnal BP pattern"
    if is_true(row.get("morning_surge_high", False)):
        return "Review morning BP control and medication timing with clinician"
    if is_true(row.get("high_variability", False)):
        return "Check measurement quality, stress, caffeine and adherence context"
    if row.get("dipping_category") == "non_dipper":
        return "Review night BP and sleep quality"
    if row.get("dipping_category") == "insufficient_sleep":
        return "Repeat or review ABPM because sleep BP data are insufficient"
    return "Routine follow-up if clinically appropriate"


def build_dryad_feature_table(
    valid_bp: pd.DataFrame,
    meta: pd.DataFrame,
    notes: pd.DataFrame,
) -> tuple[pd.DataFrame, DryadThresholds]:
    feature_rows = [summarise_participant(group) for _, group in valid_bp.groupby("ID_str")]
    features = pd.DataFrame(feature_rows).sort_values("ID_str").reset_index(drop=True)

    sbp_sd_q75 = float(features["sbp_sd"].quantile(0.75))
    morning_surge_q75 = float(features["morning_surge_sbp"].dropna().quantile(0.75))
    features["high_variability"] = features["sbp_sd"].ge(sbp_sd_q75)
    features["morning_surge_high"] = features["morning_surge_sbp"].ge(morning_surge_q75)

    features = features.merge(
        meta[
            [
                "ID_str",
                "Sex",
                "sex_label",
                "Age",
                "age_group_label",
                "BMI",
                "Caffeine (number of cups per day)",
                "Alcohol (number of units per day)",
            ]
        ],
        on="ID_str",
        how="left",
    )
    features = features.merge(
        notes[
            [
                "ID_str",
                "CGM? Yes/No",
                "Issue reported/known?",
                "known_device_issue",
                "BP cuff",
                "ECG setup",
            ]
        ],
        on="ID_str",
        how="left",
    )
    features["profile_summary"] = features.apply(profile_summary, axis=1)
    features["monitoring_recommendation"] = features.apply(monitoring_recommendation, axis=1)
    return features, DryadThresholds(sbp_sd_q75, morning_surge_q75)


def profile_summary(row: pd.Series) -> str:
    profiles = []
    dipping = row.get("dipping_category")
    if isinstance(dipping, str):
        profiles.append(dipping)
    if is_true(row.get("sustained_high_bp")):
        profiles.append("sustained_high_bp")
    if is_true(row.get("morning_surge_high", False)):
        profiles.append("high_morning_surge")
    if is_true(row.get("high_variability", False)):
        profiles.append("high_variability")
    return "; ".join(profiles)


def optional_physiology_coverage(dryad_dir: Path = DRYAD_DIR) -> pd.DataFrame:
    participant_root = dryad_dir / "Per_Participant_Sensor_Data"
    ecg_segment_root = dryad_dir / "Output_ECG_Segmentor_data"
    rows = []
    for participant_dir in sorted(p for p in participant_root.iterdir() if p.is_dir()):
        pid = participant_dir.name
        zephyr_dir = participant_dir / f"{pid}_Zephyr"
        cgm_dir = participant_dir / f"{pid}_CGM"
        abpm_dir = participant_dir / f"{pid}_ABPM"
        rows.append(
            {
                "ID_str": pid,
                "has_abpm_export": abpm_dir.exists() and any(abpm_dir.iterdir()),
                "has_cgm_file": (cgm_dir / f"{pid}_glucose.csv").exists(),
                "has_zephyr_summary": zephyr_dir.exists()
                and any(zephyr_dir.glob("*_Summary.csv")),
                "zephyr_csv_files": len(list(zephyr_dir.glob("*.csv"))) if zephyr_dir.exists() else 0,
                "has_ecg_segments": (ecg_segment_root / f"ecg_segmentsbp_{pid}.csv").exists(),
                "has_ecg_segments_bp_merged": (
                    ecg_segment_root / f"ecg_segmentsbp_{pid}BP.csv"
                ).exists(),
            }
        )
    return pd.DataFrame(rows)


def load_kaggle_arff(path: Path = KAGGLE_ARFF) -> pd.DataFrame:
    attributes: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.lower().startswith("@attribute"):
            parts = stripped.split(None, 2)
            attributes.append(parts[1].strip("'\""))
        elif stripped.lower().startswith("@data"):
            in_data = True
        elif in_data:
            data_lines.append(stripped)

    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=attributes)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def kaggle_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in KAGGLE_LABELS:
        positives = int(df[label].sum())
        rows.append(
            {
                "label": label,
                "positive": positives,
                "negative": int(len(df) - positives),
                "positive_rate": positives / len(df),
            }
        )
    return pd.DataFrame(rows)


def evaluate_classifier(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.zeros(len(y), dtype=float)
    predictions = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        if hasattr(fitted, "predict_proba"):
            probabilities[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            probabilities[test_idx] = fitted.decision_function(X.iloc[test_idx])
        predictions[test_idx] = fitted.predict(X.iloc[test_idx])

    return {
        "auroc": roc_auc_score(y, probabilities),
        "f1": f1_score(y, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
    }


def train_kaggle_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in df.columns if column not in KAGGLE_LABELS]
    X = df[feature_columns]
    model_specs = {
        "LogisticRegression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    metric_rows = []
    importance_rows = []
    for target in KAGGLE_TARGETS:
        y = df[target].astype(int)
        for model_name, model in model_specs.items():
            metrics = evaluate_classifier(model, X, y)
            metric_rows.append({"target": target, "model": model_name, **metrics})

            fitted = clone(model)
            fitted.fit(X, y)
            if model_name == "LogisticRegression":
                importances = np.abs(fitted.named_steps["model"].coef_[0])
            else:
                importances = fitted.feature_importances_
            ranking = (
                pd.DataFrame({"feature": feature_columns, "importance": importances})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            ranking["rank"] = ranking.index + 1
            ranking["target"] = target
            ranking["model"] = model_name
            importance_rows.extend(ranking.head(15).to_dict("records"))

    return pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def ensure_output_dirs(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def save_bar_labels(ax, values: Iterable[int]) -> None:
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_framework_pipeline(figure_dir: Path) -> None:
    steps = [
        "24-hour\nABPM data",
        "Sleep/wake\nseparation",
        "BP feature\nextraction",
        "Circadian BP\nprofiles",
        "Clinician-review\nrecommendation",
    ]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(steps))
    for idx, (xpos, label) in enumerate(zip(xs, steps)):
        ax.text(
            xpos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef4f8", edgecolor="#4f6f7f"),
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(xs[idx + 1] - 0.08, 0.5),
                xytext=(xpos + 0.08, 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#334e5c"),
            )
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_1_framework_pipeline.png", dpi=220)
    plt.close(fig)


def plot_dryad_dipping_categories(features: pd.DataFrame, figure_dir: Path) -> None:
    order = ["normal_dipper", "non_dipper", "reverse_dipper", "extreme_dipper", "insufficient_sleep"]
    counts = features["dipping_category"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4f8a6a", "#c49a42", "#b45a54", "#6277b8", "#8f8f8f"]
    counts.plot(kind="bar", color=colors, ax=ax)
    ax.set_title("Dryad Sleep-Aware Dipping Categories")
    ax.set_xlabel("")
    ax.set_ylabel("Participants")
    ax.set_xticklabels([label.replace("_", " ") for label in counts.index], rotation=25, ha="right")
    save_bar_labels(ax, counts.tolist())
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_2_dipping_categories.png", dpi=220)
    plt.close(fig)


def plot_awake_sleep_sbp(features: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = features.loc[features["dipping_category"].ne("insufficient_sleep")].copy()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(plot_df["awake_mean_sbp"], plot_df["sleep_mean_sbp"], s=55, color="#3b6f8f")
    for _, row in plot_df.iterrows():
        ax.text(row["awake_mean_sbp"] + 0.6, row["sleep_mean_sbp"] + 0.6, row["ID_str"], fontsize=7)
    lim_min = min(plot_df["awake_mean_sbp"].min(), plot_df["sleep_mean_sbp"].min()) - 5
    lim_max = max(plot_df["awake_mean_sbp"].max(), plot_df["sleep_mean_sbp"].max()) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", color="#6f6f6f", lw=1)
    ax.axhline(120, color="#b45a54", ls=":", lw=1.4, label="Asleep SBP 120")
    ax.axvline(135, color="#c49a42", ls=":", lw=1.4, label="Awake SBP 135")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Awake mean SBP (mmHg)")
    ax.set_ylabel("Sleep mean SBP (mmHg)")
    ax.set_title("Awake Versus Sleep SBP")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_3_awake_vs_sleep_sbp.png", dpi=220)
    plt.close(fig)


def plot_morning_surge(features: pd.DataFrame, thresholds: DryadThresholds, figure_dir: Path) -> None:
    values = features["morning_surge_sbp"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=min(10, max(4, len(values) // 2)), color="#6d8b74", edgecolor="white")
    ax.axvline(thresholds.morning_surge_q75, color="#b45a54", lw=2, label="Top quartile")
    ax.set_title("Morning Surge Distribution")
    ax.set_xlabel("Morning surge SBP (mmHg)")
    ax.set_ylabel("Participants")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_4_morning_surge_distribution.png", dpi=220)
    plt.close(fig)


def plot_example_bp_curves(valid_bp: pd.DataFrame, features: pd.DataFrame, figure_dir: Path) -> None:
    desired = ["normal_dipper", "non_dipper", "reverse_dipper", "extreme_dipper"]
    selected: list[str] = []
    titles: list[str] = []
    for category in desired:
        candidates = features.loc[features["dipping_category"].eq(category), "ID_str"].tolist()
        if candidates:
            selected.append(candidates[0])
            titles.append(category.replace("_", " "))
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 1, figsize=(9, 2.7 * len(selected)), sharex=False)
    if len(selected) == 1:
        axes = [axes]
    for ax, pid, title in zip(axes, selected, titles):
        group = valid_bp.loc[valid_bp["ID_str"].eq(pid)].sort_values("measurement_datetime")
        colors = group["Wake_Sleep"].map({1: "#3b6f8f", 0: "#6b5b95"})
        ax.plot(group["hours_since_start"], group["Systolic"], color="#a0a0a0", lw=1)
        ax.scatter(group["hours_since_start"], group["Systolic"], c=colors, s=28)
        ax.set_title(f"Participant {pid}: {title}")
        ax.set_ylabel("SBP")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Hours since first valid ABPM reading")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_5_example_bp_curves.png", dpi=220)
    plt.close(fig)


def plot_kaggle_feature_importance(feature_importance: pd.DataFrame, figure_dir: Path) -> None:
    rf = feature_importance.loc[feature_importance["model"].eq("RandomForest")]
    average = (
        rf.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(12)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(average["feature"], average["importance"], color="#4f6f7f")
    ax.set_title("Kaggle Random Forest Feature Importance")
    ax.set_xlabel("Mean importance across ABPM targets")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_6_kaggle_feature_importance.png", dpi=220)
    plt.close(fig)


def plot_clinical_pathway(figure_dir: Path) -> None:
    rows = [
        ("Normal dipper", "Routine follow-up"),
        ("Non-dipper", "Review night BP and sleep quality"),
        ("Reverse dipper", "Prioritise clinician review"),
        ("Morning surge", "Review morning BP control and timing"),
        ("High variability", "Check stress, caffeine, adherence and measurement quality"),
        ("Sustained high BP", "Early treatment-control review"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Detected profile", "Monitoring recommendation"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#eef4f8")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#cfd8dc")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_7_clinical_monitoring_pathway.png", dpi=220)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    raw_bp: pd.DataFrame,
    valid_bp: pd.DataFrame,
    features: pd.DataFrame,
    sensitivity_features: pd.DataFrame,
    coverage: pd.DataFrame,
    kaggle_labels: pd.DataFrame,
    kaggle_metrics: pd.DataFrame,
    thresholds: DryadThresholds,
) -> None:
    insufficient = features.loc[
        features["dipping_category"].eq("insufficient_sleep"), "ID_str"
    ].tolist()
    issue_excluded = features.loc[features["known_device_issue"].map(is_true), "ID_str"].tolist()
    dipping_counts = features["dipping_category"].value_counts().to_dict()
    high_variability = int(features["high_variability"].sum())
    high_morning = int(features["morning_surge_high"].sum())
    sustained = int(features["sustained_high_bp"].map(is_true).sum())

    best_metrics = (
        kaggle_metrics.sort_values(["target", "balanced_accuracy"], ascending=[True, False])
        .groupby("target")
        .head(1)
    )
    label_rows = "\n".join(
        f"- {row.label}: {row.positive} positive / {row.negative} negative "
        f"({row.positive_rate:.1%} positive)"
        for row in kaggle_labels.itertuples(index=False)
    )
    metric_rows = "\n".join(
        f"- {row.target}: {row.model}, AUROC {row.auroc:.3f}, "
        f"F1 {row.f1:.3f}, balanced accuracy {row.balanced_accuracy:.3f}"
        for row in best_metrics.itertuples(index=False)
    )

    text = f"""# Sleep-Aware Blood Pressure Profiling Framework Results

## Dryad ABPM Data Quality
- Raw sleep-annotated ABPM rows: {len(raw_bp):,}
- Valid rows after removing zero SBP/DBP/MAP/HR: {len(valid_bp):,}
- Participants with valid readings: {valid_bp['ID_str'].nunique()}
- Participants with insufficient valid sleep BP for dipping/surge: {', '.join(insufficient)}
- Participants marked with known device/data issues: {', '.join(issue_excluded)}

## Dryad Participant Profiles
- Dipping category counts: {dipping_counts}
- High SBP variability threshold: top quartile, SBP SD >= {thresholds.sbp_sd_q75:.2f} mmHg
- High morning surge threshold: top quartile, surge >= {thresholds.morning_surge_q75:.2f} mmHg
- High-variability participants: {high_variability}
- High morning-surge participants: {high_morning}
- Sustained high BP participants: {sustained}
- Sensitivity cohort without known device issues: {len(sensitivity_features)} participants

## Optional Physiological Coverage
- ABPM exports: {int(coverage['has_abpm_export'].sum())} participants
- Zephyr summary files: {int(coverage['has_zephyr_summary'].sum())} participants
- CGM files: {int(coverage['has_cgm_file'].sum())} participants
- ECG segment files: {int(coverage['has_ecg_segments'].sum())} participants
- BP-merged ECG segment files: {int(coverage['has_ecg_segments_bp_merged'].sum())} participants

## Kaggle ABPM Dataset
- Rows: {int(kaggle_labels['positive'].iloc[0] + kaggle_labels['negative'].iloc[0])}
- Targets modelled: {', '.join(KAGGLE_TARGETS)}
- `BP-Variability` was not modelled because it is positive for every row.

Label distribution:
{label_rows}

Best cross-validated model per target:
{metric_rows}

## Interpretation Boundary
The generated recommendations are clinician-review support signals. They do not provide automatic medication adjustment instructions.
"""
    (output_dir / "analysis_summary.md").write_text(text, encoding="utf-8")


def run_pipeline(
    dryad_dir: Path = DRYAD_DIR,
    kaggle_arff: Path = KAGGLE_ARFF,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    figure_dir = ensure_output_dirs(output_dir)

    raw_bp, valid_bp = load_dryad_bp(dryad_dir)
    meta, notes = load_participant_metadata(dryad_dir)
    features, thresholds = build_dryad_feature_table(valid_bp, meta, notes)

    issue_ids = set(notes.loc[notes["known_device_issue"], "ID_str"])
    sensitivity_bp = valid_bp.loc[~valid_bp["ID_str"].isin(issue_ids)].copy()
    sensitivity_features, sensitivity_thresholds = build_dryad_feature_table(
        sensitivity_bp, meta, notes
    )

    coverage = optional_physiology_coverage(dryad_dir)
    kaggle_df = load_kaggle_arff(kaggle_arff)
    labels = kaggle_label_distribution(kaggle_df)
    metrics, importance = train_kaggle_models(kaggle_df)

    features.to_csv(output_dir / "dryad_participant_features.csv", index=False)
    sensitivity_features.to_csv(
        output_dir / "dryad_participant_features_sensitivity_no_device_issues.csv",
        index=False,
    )
    valid_bp.to_csv(output_dir / "dryad_valid_bp_readings.csv", index=False)
    coverage.to_csv(output_dir / "optional_physiology_coverage.csv", index=False)
    labels.to_csv(output_dir / "kaggle_label_distribution.csv", index=False)
    metrics.to_csv(output_dir / "kaggle_model_metrics.csv", index=False)
    importance.to_csv(output_dir / "kaggle_feature_importance.csv", index=False)

    thresholds_df = pd.DataFrame(
        [
            {
                "cohort": "main",
                "sbp_sd_q75": thresholds.sbp_sd_q75,
                "morning_surge_q75": thresholds.morning_surge_q75,
            },
            {
                "cohort": "sensitivity_no_known_device_issues",
                "sbp_sd_q75": sensitivity_thresholds.sbp_sd_q75,
                "morning_surge_q75": sensitivity_thresholds.morning_surge_q75,
            },
        ]
    )
    thresholds_df.to_csv(output_dir / "dryad_thresholds.csv", index=False)

    plot_framework_pipeline(figure_dir)
    plot_dryad_dipping_categories(features, figure_dir)
    plot_awake_sleep_sbp(features, figure_dir)
    plot_morning_surge(features, thresholds, figure_dir)
    plot_example_bp_curves(valid_bp, features, figure_dir)
    plot_kaggle_feature_importance(importance, figure_dir)
    plot_clinical_pathway(figure_dir)

    write_summary(
        output_dir,
        raw_bp,
        valid_bp,
        features,
        sensitivity_features,
        coverage,
        labels,
        metrics,
        thresholds,
    )

    return {
        "features": output_dir / "dryad_participant_features.csv",
        "summary": output_dir / "analysis_summary.md",
        "metrics": output_dir / "kaggle_model_metrics.csv",
        "figures": figure_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sleep-aware BP profiling framework analysis."
    )
    parser.add_argument("--dryad-dir", type=Path, default=DRYAD_DIR)
    parser.add_argument("--kaggle-arff", type=Path, default=KAGGLE_ARFF)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    outputs = run_pipeline(args.dryad_dir, args.kaggle_arff, args.output_dir)
    print("Generated sleep-aware BP framework outputs:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
