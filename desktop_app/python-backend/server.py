"""FastAPI backend for BP Profile Monitor desktop app."""
from __future__ import annotations

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project paths so we can import the analysis modules
# server.py is at bp-profile-monitor/python-backend/server.py
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


@app.post("/api/analyse")
def analyse(req: AnalyseRequest):
    try:
        df = pd.DataFrame(req.readings)
        valid = prepare_patient_abpm(
            df, patient_id=req.patient_id,
            sleep_start=req.sleep_start, sleep_end=req.sleep_end,
        )
        profile = build_patient_profile(valid)
        return _profile_to_json(valid, profile)
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
        valid = prepare_patient_abpm(
            df, patient_id=patient_id,
            sleep_start=req.sleep_start, sleep_end=req.sleep_end,
        )
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
