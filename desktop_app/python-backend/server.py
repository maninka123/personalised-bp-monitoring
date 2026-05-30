"""FastAPI backend for BP Profile Monitor desktop app."""
from __future__ import annotations

import argparse
import base64
import math
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project paths so we can import the analysis modules.
# Works when desktop_app is inside the repo, and when it sits beside the repo.
ROOT = Path(__file__).resolve().parents[2]
PERSONALISED_CANDIDATES = [ROOT, ROOT / "personalised-bp-monitoring"]
PERSONALISED = next(
    (candidate for candidate in PERSONALISED_CANDIDATES if (candidate / "clinical_report_utils.py").exists()),
    ROOT,
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PERSONALISED))

import pandas as pd
import numpy as np

from clinical_report_utils import (
    build_patient_profile,
    create_pdf_report,
    extract_patient_details,
    prepare_patient_abpm,
    pattern_flags,
    priority_level,
    profile_label,
    review_points,
)
from sleep_aware_bp_framework import is_true

try:
    from bp_report_assistant import (
        answer_report_question,
        build_report_context,
        token_status,
    )
except Exception:  # pragma: no cover - keeps the desktop app usable if assistant deps are missing
    answer_report_question = None
    build_report_context = None
    token_status = None

NN_MODEL = None
NN_META: dict[str, Any] | None = None
NN_MODEL_ERROR: str | None = None

app = FastAPI(title="BP Profile Monitor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyseRequest(BaseModel):
    readings: list[dict]
    patient_id: str = "NEW"
    sleep_start: str = "22:00"
    sleep_end: str = "07:00"


class FileAnalyseRequest(BaseModel):
    filename: str
    data_base64: str
    patient_id: str = "NEW"
    sleep_start: str = "22:00"
    sleep_end: str = "07:00"


class ExportPdfRequest(BaseModel):
    readings: list[dict]
    patient_id: str = "NEW"
    patient_name: str = ""
    patient_age: str = ""
    patient_sex: str = ""
    patient_bmi: str = ""
    abpm_date: str = ""
    sleep_start: str = "22:00"
    sleep_end: str = "07:00"


class AskReportRequest(BaseModel):
    profile: dict[str, Any]
    question: str


class LimitedInputPredictRequest(BaseModel):
    model_variant: str = "distilled_ssl_student"
    input_scenario: str = "all_limited_inputs"
    clinic_sbp: Any = None
    clinic_dbp: Any = None
    morning_home_sbp: Any = None
    morning_home_dbp: Any = None
    evening_home_sbp: Any = None
    evening_home_dbp: Any = None
    home_3day_mean_sbp: Any = None
    home_3day_mean_dbp: Any = None
    home_7day_mean_sbp: Any = None
    home_7day_mean_dbp: Any = None
    home_bp_variability: Any = None
    age: Any = None
    sex: Any = None
    bmi: Any = None
    resting_hr: Any = None
    diabetes: Any = None
    smoker: Any = None
    previous_hypertension: Any = None
    medication_status: Any = None
    sleep_duration: Any = None
    sleep_quality: Any = None
    caffeine_cups: Any = None
    alcohol_units: Any = None
    stress_level: Any = None


def _safe(v: Any) -> Any:
    """Convert numpy/pandas types to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _num(value: Any) -> float | None:
    """Parse optional numeric input from browser forms."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"na", "n/a", "none", "null", "unknown", "missing"}:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _cat(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"na", "n/a", "none", "null", "unknown", "missing"}:
        return None
    return text


def _yes_no(value: Any) -> float | None:
    text = _cat(value)
    if text is None:
        return None
    if text in {"yes", "y", "true", "1", "current", "treated", "on medication"}:
        return 1.0
    if text in {"no", "n", "false", "0", "never", "untreated", "none"}:
        return 0.0
    if text in {"former", "past"}:
        return 0.45
    return None


def _mean_present(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sigmoid(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


def _round_prob(value: float) -> float:
    return round(_clip(value, 0.01, 0.99), 3)


def _feature_label(name: str) -> str:
    return name.replace("_", " ").title()


def _limited_input_prediction(req: LimitedInputPredictRequest) -> dict[str, Any]:
    """Prototype missingness-aware limited-input ABPM risk estimator.

    This is intentionally transparent and conservative. It implements the app-facing
    proof-of-concept for the teacher-student method before a fully trained model is
    produced from paired limited-input + ABPM data.
    """
    numeric_names = [
        "clinic_sbp", "clinic_dbp", "morning_home_sbp", "morning_home_dbp",
        "evening_home_sbp", "evening_home_dbp", "home_3day_mean_sbp",
        "home_3day_mean_dbp", "home_7day_mean_sbp", "home_7day_mean_dbp",
        "home_bp_variability", "age", "bmi", "resting_hr", "sleep_duration",
        "caffeine_cups", "alcohol_units", "stress_level",
    ]
    categorical_names = [
        "sex", "diabetes", "smoker", "previous_hypertension",
        "medication_status", "sleep_quality",
    ]
    values = {name: _num(getattr(req, name)) for name in numeric_names}
    cats = {name: _cat(getattr(req, name)) for name in categorical_names}

    missing_features = [
        _feature_label(name) for name, value in {**values, **cats}.items() if value is None
    ]
    present_features = [
        _feature_label(name) for name, value in {**values, **cats}.items() if value is not None
    ]

    clinic_sbp = values["clinic_sbp"]
    clinic_dbp = values["clinic_dbp"]
    morning_sbp = values["morning_home_sbp"]
    morning_dbp = values["morning_home_dbp"]
    evening_sbp = values["evening_home_sbp"]
    evening_dbp = values["evening_home_dbp"]
    home3_sbp = values["home_3day_mean_sbp"]
    home3_dbp = values["home_3day_mean_dbp"]
    home7_sbp = values["home_7day_mean_sbp"]
    home7_dbp = values["home_7day_mean_dbp"]

    home_sbp = _mean_present([morning_sbp, evening_sbp, home3_sbp, home7_sbp])
    home_dbp = _mean_present([morning_dbp, evening_dbp, home3_dbp, home7_dbp])
    estimated_awake_sbp = _mean_present([home_sbp, clinic_sbp])
    estimated_awake_dbp = _mean_present([home_dbp, clinic_dbp])
    if estimated_awake_sbp is None:
        estimated_awake_sbp = 132.0
    if estimated_awake_dbp is None:
        estimated_awake_dbp = 82.0

    age = values["age"] if values["age"] is not None else 52.0
    bmi = values["bmi"] if values["bmi"] is not None else 27.0
    resting_hr = values["resting_hr"] if values["resting_hr"] is not None else 74.0
    sleep_duration = values["sleep_duration"] if values["sleep_duration"] is not None else 7.0
    stress = values["stress_level"] if values["stress_level"] is not None else 3.0
    caffeine = values["caffeine_cups"] if values["caffeine_cups"] is not None else 1.5

    diabetes = _yes_no(cats["diabetes"]) or 0.0
    smoker = _yes_no(cats["smoker"]) or 0.0
    previous_htn = _yes_no(cats["previous_hypertension"]) or 0.0
    medication = _yes_no(cats["medication_status"]) or 0.0
    poor_sleep = 0.0
    if cats["sleep_quality"] in {"poor", "very poor", "fragmented"}:
        poor_sleep = 1.0
    elif cats["sleep_quality"] in {"fair", "mixed"}:
        poor_sleep = 0.45

    home_available = home_sbp is not None or home_dbp is not None
    clinic_available = clinic_sbp is not None or clinic_dbp is not None
    demographics_available = any(values[k] is not None for k in ("age", "bmi")) or cats["sex"] is not None
    history_available = any(cats[k] is not None for k in ("diabetes", "smoker", "previous_hypertension", "medication_status"))
    lifestyle_available = any(values[k] is not None for k in ("sleep_duration", "caffeine_cups", "alcohol_units", "stress_level")) or cats["sleep_quality"] is not None
    present_groups = [
        name for name, ok in {
            "clinic BP": clinic_available,
            "home BP": home_available,
            "demographics": demographics_available,
            "history": history_available,
            "lifestyle": lifestyle_available,
        }.items() if ok
    ]

    completeness = (len(present_features) / max(1, len(numeric_names) + len(categorical_names)))
    group_completeness = len(present_groups) / 5.0
    missing_load = 1.0 - (0.55 * completeness + 0.45 * group_completeness)

    avg_sbp_for_load = _mean_present([home_sbp, clinic_sbp]) or estimated_awake_sbp
    avg_dbp_for_load = _mean_present([home_dbp, clinic_dbp]) or estimated_awake_dbp
    sbp_load = max(0.0, (avg_sbp_for_load - 125.0) / 15.0)
    dbp_load = max(0.0, (avg_dbp_for_load - 80.0) / 10.0)
    clinic_high = max(0.0, ((clinic_sbp if clinic_sbp is not None else estimated_awake_sbp) - 135.0) / 18.0)
    home_high = max(0.0, ((home_sbp if home_sbp is not None else estimated_awake_sbp - 3.0) - 130.0) / 16.0)
    age_term = max(0.0, (age - 50.0) / 25.0)
    bmi_term = max(0.0, (bmi - 25.0) / 8.0)
    hr_term = max(0.0, (resting_hr - 75.0) / 20.0)
    stress_term = _clip((stress - 1.0) / 4.0, 0.0, 1.0)
    caffeine_term = _clip(caffeine / 4.0, 0.0, 1.0)
    short_sleep_term = max(0.0, (6.5 - sleep_duration) / 2.0)

    morning_evening_gap = 0.0
    if morning_sbp is not None and evening_sbp is not None:
        morning_evening_gap = (morning_sbp - evening_sbp) / 12.0
    morning_high = max(0.0, ((morning_sbp if morning_sbp is not None else estimated_awake_sbp) - 135.0) / 16.0)

    clinic_home_gap = 0.0
    if clinic_sbp is not None and home_sbp is not None:
        clinic_home_gap = abs(clinic_sbp - home_sbp) / 15.0
    variability_input = values["home_bp_variability"]
    if variability_input is None:
        variability_input = abs((morning_sbp or estimated_awake_sbp) - (evening_sbp or estimated_awake_sbp)) * 0.55
        variability_input += clinic_home_gap * 4.0
    variability_term = _clip(variability_input / 13.0, 0.0, 2.0)

    abnormal_dipping = _sigmoid(
        -0.65 + 0.38 * sbp_load + 0.25 * age_term + 0.25 * bmi_term
        + 0.35 * diabetes + 0.32 * poor_sleep + 0.26 * short_sleep_term
        + 0.22 * stress_term + 0.18 * previous_htn + 0.16 * medication
        + 0.12 * missing_load
    )
    morning_surge = _sigmoid(
        -0.9 + 0.72 * morning_evening_gap + 0.42 * morning_high
        + 0.24 * age_term + 0.24 * stress_term + 0.18 * caffeine_term
        + 0.16 * smoker + 0.16 * clinic_high + 0.10 * hr_term
    )
    nocturnal_hypertension = _sigmoid(
        -0.72 + 0.55 * home_high + 0.42 * clinic_high + 0.28 * bmi_term
        + 0.30 * diabetes + 0.25 * poor_sleep + 0.18 * previous_htn
        + 0.14 * medication + 0.12 * missing_load
    )
    high_bp_burden = _sigmoid(
        -1.05 + 0.72 * sbp_load + 0.42 * dbp_load + 0.22 * previous_htn
        + 0.18 * diabetes + 0.12 * medication + 0.12 * age_term
    )
    high_variability = _sigmoid(
        -1.05 + 0.58 * variability_term + 0.32 * stress_term
        + 0.23 * caffeine_term + 0.20 * smoker + 0.22 * clinic_home_gap
        + 0.10 * missing_load
    )

    model_variant = _cat(req.model_variant) or "distilled_ssl_student"
    if model_variant == "clinic_baseline":
        calibration = 0.82
        missing_penalty = 18
        model_label = "Clinic-only baseline"
    elif model_variant == "student_masks":
        calibration = 0.93
        missing_penalty = 10
        model_label = "Student with missingness masks"
    else:
        calibration = 0.98
        missing_penalty = 6
        model_label = "ABPM-TSL distilled student + SSL teacher"

    risks_raw = {
        "abnormal_dipping": abnormal_dipping,
        "morning_surge_high": morning_surge,
        "nocturnal_hypertension": nocturnal_hypertension,
        "high_bp_burden": high_bp_burden,
        "high_variability": high_variability,
    }
    risks = {
        key: _round_prob(0.5 + (value - 0.5) * calibration)
        for key, value in risks_raw.items()
    }

    sorted_risks = sorted(risks.values(), reverse=True)
    priority_score = 0.50 * sorted_risks[0] + 0.30 * (sum(sorted_risks[:3]) / 3.0) + 0.20 * risks["high_bp_burden"]
    confidence_score = _clip(32 + 68 * (0.55 * group_completeness + 0.45 * completeness) - missing_penalty * missing_load, 8, 96)
    if confidence_score < 32 and priority_score < 0.58:
        priority_label = "Data insufficient - collect more inputs"
        priority_level = "grey"
    elif priority_score >= 0.70:
        priority_label = "High ABPM review priority"
        priority_level = "red"
    elif priority_score >= 0.48:
        priority_label = "Review soon / consider ABPM"
        priority_level = "yellow"
    else:
        priority_label = "Routine monitoring"
        priority_level = "green"

    estimated_dipping_pct = _clip(16 - risks["abnormal_dipping"] * 19 - risks["nocturnal_hypertension"] * 5, -8, 24)
    sleep_sbp_proxy = estimated_awake_sbp * (1 - estimated_dipping_pct / 100.0)
    surge_proxy = _clip(
        ((morning_sbp if morning_sbp is not None else estimated_awake_sbp + 3) - sleep_sbp_proxy),
        0,
        45,
    )
    burden_score = _clip((risks["high_bp_burden"] * 100), 0, 100)
    variability_score = _clip((risks["high_variability"] * 100), 0, 100)

    driver_candidates = [
        ("High BP burden in available clinic/home inputs", sbp_load + dbp_load),
        ("Morning BP higher than evening/home baseline", max(0.0, morning_evening_gap)),
        ("Clinic-home BP mismatch", clinic_home_gap),
        ("Poor or short sleep signal", poor_sleep + short_sleep_term),
        ("Stress/caffeine variability signal", stress_term + caffeine_term),
        ("Age/BMI/metabolic risk signal", age_term + bmi_term + diabetes),
        ("Missing inputs reduced certainty", missing_load),
    ]
    drivers = [label for label, score in sorted(driver_candidates, key=lambda x: x[1], reverse=True) if score > 0.05][:4]

    recommendations = []
    if not home_available:
        recommendations.append("Collect morning and evening home BP if possible; it has the largest effect on limited-input confidence.")
    if not lifestyle_available:
        recommendations.append("Add sleep duration, sleep quality, caffeine, alcohol, and stress fields to improve missingness-aware estimates.")
    if risks["morning_surge_high"] >= 0.55:
        recommendations.append("Prioritise ABPM or structured morning home BP review because morning surge risk is elevated.")
    if risks["nocturnal_hypertension"] >= 0.55 or risks["abnormal_dipping"] >= 0.55:
        recommendations.append("Use ABPM confirmation when feasible because night-time patterns cannot be diagnosed from limited inputs alone.")
    if not recommendations:
        recommendations.append("Continue routine BP review and use ABPM when clinical concern remains.")

    return {
        "model": model_label,
        "model_variant": model_variant,
        "input_scenario": req.input_scenario,
        "model_status": "Prototype method-development estimator; not externally validated for diagnosis.",
        "risk_probabilities": risks,
        "priority": {
            "label": priority_label,
            "level": priority_level,
            "score": round(priority_score, 3),
        },
        "confidence": {
            "score": round(confidence_score, 1),
            "label": "High" if confidence_score >= 75 else "Moderate" if confidence_score >= 50 else "Limited" if confidence_score >= 30 else "Very limited",
            "input_completeness": round(completeness, 3),
            "present_groups": present_groups,
        },
        "estimated_targets": {
            "dipping_percentage_proxy": round(estimated_dipping_pct, 1),
            "sleep_mean_sbp_proxy": round(sleep_sbp_proxy, 1),
            "morning_surge_mmhg_proxy": round(surge_proxy, 1),
            "bp_burden_score": round(burden_score, 1),
            "variability_score": round(variability_score, 1),
        },
        "missingness": {
            "missing_features": missing_features,
            "present_features": present_features,
            "missing_count": len(missing_features),
            "present_count": len(present_features),
        },
        "explanation": drivers,
        "recommendations": recommendations,
        "clinical_boundary": (
            "This tool estimates ABPM-defined risk patterns for prioritisation only. "
            "It does not diagnose hypertension and does not replace ABPM."
        ),
    }


def _nn_model_paths() -> tuple[Path, Path]:
    executable_resource_models = None
    try:
        executable_resource_models = Path(sys.executable).resolve().parents[2] / "ABPM-TSL" / "models"
    except Exception:
        executable_resource_models = None
    candidates = [
        ROOT / "ABPM-TSL" / "models",
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parent.parent / "ABPM-TSL" / "models",
    ]
    if executable_resource_models is not None:
        candidates.append(executable_resource_models)
    for folder in candidates:
        model_path = folder / "student_abpm_tsl_torchscript.pt"
        meta_path = folder / "preprocessing.json"
        if model_path.exists() and meta_path.exists():
            return model_path, meta_path
    return candidates[0] / "student_abpm_tsl_torchscript.pt", candidates[0] / "preprocessing.json"


def _load_nn_model() -> bool:
    global NN_MODEL, NN_META, NN_MODEL_ERROR
    if NN_MODEL is not None and NN_META is not None:
        return True
    model_path, meta_path = _nn_model_paths()
    try:
        import json
        import torch
        NN_MODEL = torch.jit.load(str(model_path), map_location="cpu")
        NN_MODEL.eval()
        NN_META = json.loads(meta_path.read_text(encoding="utf-8"))
        NN_MODEL_ERROR = None
        return True
    except Exception as exc:
        NN_MODEL = None
        NN_META = None
        NN_MODEL_ERROR = str(exc)
        return False


def _nn_feature_value(req: LimitedInputPredictRequest, feature: str) -> tuple[float | None, float]:
    if feature in {
        "clinic_sbp", "clinic_dbp", "morning_home_sbp", "morning_home_dbp",
        "evening_home_sbp", "evening_home_dbp", "home_3day_mean_sbp",
        "home_3day_mean_dbp", "home_7day_mean_sbp", "home_7day_mean_dbp",
        "home_bp_variability", "age", "bmi", "resting_hr", "sleep_duration",
        "caffeine_cups", "alcohol_units", "stress_level",
    }:
        value = _num(getattr(req, feature))
        return value, 1.0 if value is not None else 0.0
    if feature == "sex_female":
        sex = _cat(req.sex)
        if sex is None:
            return None, 0.0
        return (1.0 if sex.startswith("f") else 0.0), 1.0
    if feature == "sleep_quality_ord":
        quality = _cat(req.sleep_quality)
        if quality is None:
            return None, 0.0
        if quality in {"good", "very good"}:
            return 3.0, 1.0
        if quality == "fair":
            return 2.0, 1.0
        if quality in {"poor", "very poor", "fragmented"}:
            return 1.0, 1.0
        return None, 0.0
    if feature == "diabetes":
        value = _yes_no(req.diabetes)
    elif feature == "smoker":
        value = _yes_no(req.smoker)
    elif feature == "previous_hypertension":
        value = _yes_no(req.previous_hypertension)
    elif feature == "medication_status":
        value = _yes_no(req.medication_status)
    else:
        value = None
    return value, 1.0 if value is not None else 0.0


def _limited_input_nn_prediction(req: LimitedInputPredictRequest) -> dict[str, Any] | None:
    if not _load_nn_model() or NN_META is None or NN_MODEL is None:
        return None
    try:
        import torch
        feature_names = NN_META["feature_names"]
        means = np.array(NN_META["feature_mean"], dtype="float32")
        stds = np.array(NN_META["feature_std"], dtype="float32")
        reg_mean = np.array(NN_META["reg_mean"], dtype="float32")
        reg_std = np.array(NN_META["reg_std"], dtype="float32")
        values = []
        masks = []
        missing_features = []
        present_features = []
        for idx, feature in enumerate(feature_names):
            value, mask = _nn_feature_value(req, feature)
            masks.append(mask)
            if mask:
                values.append(value)
                present_features.append(_feature_label(feature))
            else:
                values.append(float(means[idx]))
                missing_features.append(_feature_label(feature))
        x = ((np.array(values, dtype="float32") - means) / stds).reshape(1, -1)
        mask_arr = np.array(masks, dtype="float32").reshape(1, -1)
        with torch.no_grad():
            probs_t, reg_t = NN_MODEL(torch.tensor(x), torch.tensor(mask_arr))
        probs = probs_t.numpy()[0]
        reg_values = reg_t.numpy()[0] * reg_std + reg_mean
        risks = {name: _round_prob(float(probs[i])) for i, name in enumerate(NN_META["target_names"])}
        priority_score = (
            0.45 * max(risks.values())
            + 0.30 * float(np.mean(sorted(risks.values(), reverse=True)[:3]))
            + 0.25 * risks.get("high_bp_burden", 0.0)
        )
        if priority_score >= 0.70:
            priority_label, priority_level = "High ABPM review priority", "red"
        elif priority_score >= 0.48:
            priority_label, priority_level = "Review soon / consider ABPM", "yellow"
        else:
            priority_label, priority_level = "Routine monitoring", "green"
        groups = {
            "clinic BP": any(mask_arr[0, feature_names.index(f)] for f in ["clinic_sbp", "clinic_dbp"]),
            "home BP": any(mask_arr[0, feature_names.index(f)] for f in [
                "morning_home_sbp", "evening_home_sbp", "home_3day_mean_sbp", "home_7day_mean_sbp"
            ]),
            "demographics": any(mask_arr[0, feature_names.index(f)] for f in ["age", "sex_female", "bmi"]),
            "history": any(mask_arr[0, feature_names.index(f)] for f in ["diabetes", "smoker", "previous_hypertension", "medication_status"]),
            "lifestyle": any(mask_arr[0, feature_names.index(f)] for f in ["sleep_duration", "sleep_quality_ord", "caffeine_cups", "alcohol_units", "stress_level"]),
        }
        present_groups = [name for name, ok in groups.items() if ok]
        completeness = float(mask_arr.mean())
        confidence_score = _clip(25 + 75 * (0.55 * completeness + 0.45 * len(present_groups) / 5.0), 5, 96)
        top_risks = sorted(risks.items(), key=lambda item: item[1], reverse=True)[:3]
        recommendations = []
        if not groups["home BP"]:
            recommendations.append("Collect morning and evening home BP to improve limited-input confidence.")
        if risks.get("nocturnal_hypertension", 0) >= 0.55 or risks.get("abnormal_dipping", 0) >= 0.55:
            recommendations.append("Use ABPM confirmation where feasible because sleep BP patterns cannot be diagnosed from limited inputs alone.")
        if risks.get("morning_surge_high", 0) >= 0.55:
            recommendations.append("Review morning BP pattern with structured home readings or ABPM.")
        if not recommendations:
            recommendations.append("Continue routine review and use ABPM if clinical concern remains.")
        return {
            "model": "ABPM-TSL trained neural student",
            "model_variant": "torchscript_student",
            "input_scenario": req.input_scenario,
            "model_status": "Trained on synthetic limited-input data derived from full ABPM; method-development only.",
            "risk_probabilities": risks,
            "priority": {"label": priority_label, "level": priority_level, "score": round(priority_score, 3)},
            "confidence": {
                "score": round(confidence_score, 1),
                "label": "High" if confidence_score >= 75 else "Moderate" if confidence_score >= 50 else "Limited",
                "input_completeness": round(completeness, 3),
                "present_groups": present_groups,
            },
            "estimated_targets": {
                "dipping_percentage_proxy": round(float(reg_values[0]), 1),
                "morning_surge_mmhg_proxy": round(float(reg_values[1]), 1),
                "sleep_mean_sbp_proxy": round(float(reg_values[2]), 1),
                "bp_burden_score": round(float(reg_values[3]), 1),
                "variability_score": round(float(reg_values[4]), 1),
            },
            "missingness": {
                "missing_features": missing_features,
                "present_features": present_features,
                "missing_count": len(missing_features),
                "present_count": len(present_features),
            },
            "explanation": [f"{_feature_label(name)} risk probability {prob:.0%}" for name, prob in top_risks],
            "recommendations": recommendations,
            "clinical_boundary": (
                "This neural model estimates ABPM-defined risk patterns for prioritisation only. "
                "It does not diagnose hypertension and does not replace ABPM."
            ),
        }
    except Exception as exc:
        global NN_MODEL_ERROR
        NN_MODEL_ERROR = str(exc)
        return None

def _profile_to_json(
    valid: pd.DataFrame,
    profile: dict,
    patient_details: dict[str, str] | None = None,
) -> dict:
    """Convert profile dict + readings to a JSON-serializable dict."""
    out = {}
    for k, v in profile.items():
        if k in ("pattern_flags", "review_points"):
            out[k] = v
        else:
            out[k] = _safe(v)

    # Derived clinical fields for consultation view
    out["profile_label"] = profile_label(profile)
    out["priority_level"] = priority_level(profile.get("priority", ""))
    out["patient_explanation"] = profile.get("patient_explanation", "")
    out["data_quality"] = profile.get("data_quality", "")

    # Dipping info for day-vs-night visual
    dip_pct = _safe(profile.get("dipping_pct_sbp"))
    out["dipping_pct_sbp"] = dip_pct
    out["dipping_observed_text"] = (
        f"{abs(dip_pct):.1f}% {'fall' if dip_pct and dip_pct > 0 else 'rise'}"
        if dip_pct is not None else "N/A"
    )

    # Curve caption
    captions = []
    if profile.get("dipping_category") in ("non_dipper", "reverse_dipper"):
        captions.append("BP stayed high during sleep")
    if profile.get("morning_surge_high"):
        captions.append("rose further after waking")
    if is_true(profile.get("sustained_high_bp")):
        captions.append("BP was high across the 24-hour recording")
    out["curve_caption"] = ", and ".join(captions) + "." if captions else "No major pattern detected."

    # Clinical status table (combined flags + explanations)
    out["clinical_status_table"] = _build_clinical_table(profile)
    out["patient_details"] = patient_details or {}

    # Readings for charts
    readings = []
    for _, row in valid.head(100).iterrows():
        readings.append({
            "hours_since_start": _safe(row.get("hours_since_start")),
            "Systolic": _safe(row.get("Systolic")),
            "Diastolic": _safe(row.get("Diastolic")),
            "HR": _safe(row.get("HR")),
            "Wake_Sleep": _safe(row.get("Wake_Sleep")),
            "time_display": row["measurement_datetime"].strftime("%d %b %H:%M")
            if pd.notna(row.get("measurement_datetime"))
            else "",
        })
    out["readings"] = readings
    return out


def _build_clinical_table(profile: dict) -> list[dict]:
    """Build a doctor-friendly combined status table."""
    rows = []
    dipping = profile.get("dipping_category")

    # Sleep BP fall
    if dipping in ("non_dipper", "reverse_dipper"):
        rows.append({"pattern": "Sleep BP fall", "status": "Needs review",
                      "why": "BP did not fall enough during sleep",
                      "review": "Review night BP and sleep quality"})
    else:
        rows.append({"pattern": "Sleep BP fall", "status": "Normal",
                      "why": "BP dipped within expected range during sleep",
                      "review": "Routine review"})

    # Morning rise
    if profile.get("morning_surge_high"):
        rows.append({"pattern": "Morning rise", "status": "Needs review",
                      "why": "BP increased after waking",
                      "review": "Review morning BP control"})
    else:
        rows.append({"pattern": "Morning rise", "status": "Normal",
                      "why": "Morning BP rise within expected range",
                      "review": "Routine review"})

    # BP burden
    if is_true(profile.get("sustained_high_bp")):
        rows.append({"pattern": "BP burden", "status": "Needs review",
                      "why": "BP stayed high across monitoring",
                      "review": "Consider earlier treatment review"})
    else:
        rows.append({"pattern": "BP burden", "status": "Normal",
                      "why": "24h BP average within expected range",
                      "review": "Routine review"})

    # Pressure gap (pulse pressure)
    pp = profile.get("mean_24h_sbp", 0) - profile.get("mean_24h_dbp", 0)
    if pp and pp > 60:
        rows.append({"pattern": "Pressure gap", "status": "Needs review",
                      "why": "Wide gap between top and bottom numbers",
                      "review": "Review arterial stiffness indicators"})
    else:
        rows.append({"pattern": "Pressure gap", "status": "Within range",
                      "why": "Gap between top and bottom numbers is normal",
                      "review": "Routine review"})

    # BP variability
    if profile.get("high_variability"):
        rows.append({"pattern": "BP variability", "status": "Needs review",
                      "why": "Readings changed more than expected",
                      "review": "Check stress, caffeine, adherence"})
    else:
        rows.append({"pattern": "BP variability", "status": "Normal",
                      "why": "Readings were within expected variation",
                      "review": "Routine review"})

    return rows


def _assistant_context(profile: dict[str, Any]) -> dict[str, Any]:
    """Create a compact report summary for Gemma without sending raw readings."""
    if build_report_context is not None:
        try:
            return build_report_context(profile)
        except Exception:
            pass

    points = []
    for item in profile.get("review_points", []) or []:
        if isinstance(item, dict):
            points.append(item.get("Doctor review point") or item.get("review") or "")
        else:
            points.append(str(item))
    points = [p for p in points if p]
    if not points:
        points = [row.get("review", "") for row in profile.get("clinical_status_table", []) if row.get("status") == "Needs review"]

    return {
        "profile": profile.get("profile_label") or pretty_category(profile.get("dipping_category")),
        "priority": profile.get("priority"),
        "data_quality": profile.get("data_quality"),
        "valid_readings": profile.get("valid_readings"),
        "sleep_readings": profile.get("sleep_valid_readings"),
        "mean_24h_bp": _bp_text(profile.get("mean_24h_sbp"), profile.get("mean_24h_dbp")),
        "awake_bp": _bp_text(profile.get("awake_mean_sbp"), profile.get("awake_mean_dbp")),
        "sleep_bp": _bp_text(profile.get("sleep_mean_sbp"), profile.get("sleep_mean_dbp")),
        "dipping_percentage": _unit_text(profile.get("dipping_pct_sbp"), "%", 1),
        "dipping_category": str(profile.get("dipping_category", "")).replace("_", " "),
        "morning_surge": _unit_text(profile.get("morning_surge_sbp"), " mmHg", 0),
        "bp_variability": "High" if profile.get("high_variability") else "Not flagged",
        "sustained_high_bp": "Yes" if is_true(profile.get("sustained_high_bp")) else "No",
        "review_points": points,
        "clinical_boundary": (
            "This is clinician-review support only. It does not diagnose, prescribe, "
            "or recommend medication changes."
        ),
    }


def _bp_text(sbp: Any, dbp: Any) -> str:
    if sbp is None or dbp is None:
        return "N/A"
    try:
        if pd.isna(sbp) or pd.isna(dbp):
            return "N/A"
        return f"{float(sbp):.0f}/{float(dbp):.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _unit_text(value: Any, unit: str, decimals: int) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}{unit}"
    except (TypeError, ValueError):
        return "N/A"


def pretty_category(v: Any) -> str:
    return str(v or "Unclassified").replace("_", " ").title()



@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/assistant-status")
def assistant_status():
    status = "not configured"
    if token_status is not None:
        status = token_status().get("Hugging Face Gemma 4", "not configured")
    return {
        "assistant": "Ask About This BP Report",
        "model": "Hugging Face Gemma 4",
        "token_status": status,
        "raw_data_sent": False,
    }


@app.post("/api/limited-input-predict")
def limited_input_predict(req: LimitedInputPredictRequest):
    try:
        nn_result = _limited_input_nn_prediction(req)
        if nn_result is not None:
            return nn_result
        result = _limited_input_prediction(req)
        if NN_MODEL_ERROR:
            result["model_status"] += f" Neural model fallback reason: {NN_MODEL_ERROR}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/analyse")
def analyse(req: AnalyseRequest):
    try:
        df = pd.DataFrame(req.readings)
        valid = prepare_patient_abpm(
            df, patient_id=req.patient_id,
            sleep_start=req.sleep_start, sleep_end=req.sleep_end,
        )
        profile = build_patient_profile(valid)
        
        details = None
        if req.patient_id == "EXAMPLE":
            details = {
                "Patient ID": "SP009",
                "Patient Name": "Sample Patient Borderline",
                "Age": "41",
                "Sex": "Female",
                "BMI": "23.7"
            }
            
        return _profile_to_json(valid, profile, details)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/analyse-file")
def analyse_file(req: FileAnalyseRequest):
    try:
        raw_bytes = base64.b64decode(req.data_base64)
        name_lower = req.filename.lower()
        if name_lower.endswith(".csv"):
            df = pd.read_csv(BytesIO(raw_bytes))
        elif name_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(raw_bytes))
        else:
            raise ValueError(f"Unsupported file type: {req.filename}")
        fallback_details = {
            "Patient ID": req.patient_id,
            "Patient Name": "",
            "Age": "",
            "Sex": "",
            "BMI": "",
            "ABPM date": "",
        }
        patient_details = extract_patient_details(df, fallback_details)
        patient_id = patient_details.get("Patient ID") or req.patient_id
        try:
            valid = prepare_patient_abpm(
                df, patient_id=patient_id,
                sleep_start=req.sleep_start, sleep_end=req.sleep_end,
            )
        except Exception as e:
            if "No valid ABPM readings found" in str(e):
                cols = list(df.columns)
                head = df.head(1).to_dict()
                raise ValueError(f"No valid readings. df shape: {df.shape}. Columns: {cols}. Data: {head}")
            raise e
        profile = build_patient_profile(valid)
        return _profile_to_json(valid, profile, patient_details)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/export-pdf")
def export_pdf(req: ExportPdfRequest):
    try:
        df = pd.DataFrame(req.readings)
        valid = prepare_patient_abpm(
            df, patient_id=req.patient_id,
            sleep_start=req.sleep_start, sleep_end=req.sleep_end,
        )
        profile = build_patient_profile(valid)
        patient_details = {
            "Patient ID": req.patient_id,
            "Patient Name": req.patient_name,
            "Age": req.patient_age,
            "Sex": req.patient_sex,
            "BMI": req.patient_bmi,
            "ABPM date": req.abpm_date,
        }
        pdf_bytes = create_pdf_report(valid, profile, patient_details)
        return {"pdf_base64": base64.b64encode(pdf_bytes).decode("ascii")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ask-report")
def ask_report(req: AskReportRequest):
    if answer_report_question is None:
        raise HTTPException(status_code=503, detail="Gemma assistant module is not available.")
    try:
        context = _assistant_context(req.profile)
        response = answer_report_question(req.question, context)
        return {
            "answer": response.answer,
            "source": response.source,
            "model": "Hugging Face Gemma 4",
            "raw_data_sent": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18347)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
