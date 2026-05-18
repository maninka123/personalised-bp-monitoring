from __future__ import annotations

from datetime import date, time as dt_time
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from bp_report_assistant import (
    DEFAULT_HF_MODEL,
    answer_report_question,
    build_report_context,
    quick_questions,
    token_status,
)
from clinical_report_utils import (
    build_patient_profile,
    create_pdf_report,
    example_patient_abpm,
    extract_patient_details,
    feature_table,
    pattern_flags,
    plot_24h_bp_curve,
    plot_awake_sleep_bar,
    plot_profile_position,
    prepare_patient_abpm,
    priority_level,
    profile_label,
    review_points,
)


st.set_page_config(
    page_title="BP Profile Monitor",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — uses inline styles on cards to avoid Streamlit HTML scoping
# issues that caused raw HTML to appear in the dashboard.
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
/* sidebar */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #1a2332 0%, #243447 100%);}
[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stTimeInput label {color: #94a3b8 !important;}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] input {
    color: #0f172a !important;
    background-color: #ffffff !important;
    caret-color: #0f172a !important;
}
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #0f172a !important;
    fill: #0f172a !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSidebar"] [data-baseweb="select"] [aria-selected],
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] input[readonly] {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg,
[data-testid="stSidebar"] [data-baseweb="select"] svg * {
    fill: #0f172a !important;
    color: #0f172a !important;
}
[data-testid="stSidebar"] hr {border-color: #334155 !important;}
/* patient note */
.patient-note {
    border-left: 6px solid #2f5d7c; background: #f0f7fb;
    border-radius: 10px; padding: 1.1rem 1.2rem;
    font-size: 1.04rem; line-height: 1.6;
}
.safety-box {
    border: 1px solid #e2e8f0; background: #fffdf7;
    border-radius: 10px; padding: 1.1rem 1.2rem;
}
/* header */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    color: white; padding: 1.2rem 1.5rem; border-radius: 12px;
    margin-bottom: 1.2rem;
}
.app-header h1 {margin: 0; font-size: 1.6rem; font-weight: 700;}
.app-header p  {margin: 0.2rem 0 0; font-size: 0.92rem; opacity: 0.85;}
/* traffic light */
.traffic-banner {
    display: flex; align-items: stretch; border-radius: 12px;
    overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    margin-bottom: 1rem; border: 1px solid #e2e8f0;
}
.traffic-stripe { width: 8px; flex-shrink: 0; }
.traffic-stripe.green  { background: #22c55e; }
.traffic-stripe.yellow { background: #eab308; }
.traffic-stripe.red    { background: #ef4444; }
.traffic-stripe.grey   { background: #94a3b8; }
.traffic-body {
    flex: 1; background: #fff; padding: 1.1rem 1.4rem;
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; align-items: center;
}
.tb-profile { font-size: 1.15rem; font-weight: 700; color: #0f172a; }
.tb-priority {
    display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
}
.tb-priority.green  { background: #dcfce7; color: #15803d; }
.tb-priority.yellow { background: #fef9c3; color: #a16207; }
.tb-priority.red    { background: #fee2e2; color: #b91c1c; }
.tb-priority.grey   { background: #f1f5f9; color: #64748b; }
.tb-issue { width: 100%; font-size: 0.88rem; color: #475569; line-height: 1.55; }
.tb-issue strong { color: #0f172a; }
/* dipping visual */
.dip-flow { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0; }
.dip-box {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
    padding: 0.7rem 1rem; text-align: center; min-width: 110px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}
.dip-box .label { font-size: 0.72rem; color: #475569; font-weight: 750; text-transform: uppercase; }
.dip-box .val   { font-size: 1.3rem; font-weight: 800; color: #020617; margin-top: 0.1rem; }
.dip-arrow { font-size: 1.3rem; color: #475569; font-weight: 700; }
.reading-card {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
    padding: 0.9rem 1rem; margin-top: 0.25rem; box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    color: #0f172a; line-height: 1.42; font-size: 0.92rem;
}
.reading-card p { margin: 0 0 0.48rem 0; }
.reading-card p:last-child { margin-bottom: 0; }
.reading-card strong { color: #020617; font-weight: 800; }
.reading-card .axis-note { color: #475569; font-size: 0.82rem; margin-top: 0.65rem; }
.manual-status {
    background: #ecfdf5; border: 1px solid #86efac; color: #14532d;
    border-radius: 10px; padding: 0.65rem 0.75rem; margin: 0.55rem 0;
    font-size: 0.84rem; line-height: 1.45; font-weight: 600;
}
.manual-status.warn {
    background: #fffbeb; border-color: #fde68a; color: #92400e;
}
.manual-last {
    background: #f8fafc; border: 1px solid #cbd5e1; color: #0f172a;
    border-radius: 10px; padding: 0.6rem 0.7rem; margin: 0.45rem 0;
    font-size: 0.82rem; line-height: 1.45;
}
/* patient summary card */
.patient-summary-card {
    background: linear-gradient(135deg, #f0f7ff, #f8fafc);
    border: 1px solid #dbeafe; border-radius: 12px;
    padding: 1.3rem 1.5rem; margin: 1rem 0;
}
.patient-summary-card h3 { font-size: 1.05rem; font-weight: 700; margin: 0 0 0.5rem; }
.patient-summary-card .summary-text { font-size: 0.92rem; color: #1e293b; line-height: 1.58; }
/* curve caption */
.curve-caption {
    font-size: 0.85rem; color: #475569; padding: 0.5rem 0.8rem;
    background: #fffbeb; border-left: 3px solid #fbbf24;
    border-radius: 0 6px 6px 0; margin-top: 0.5rem;
}
.curve-caption.ok { background: #f0fdf4; border-left-color: #22c55e; }
</style>
"""


def main() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # App header
    st.markdown(
        '<div class="app-header">'
        "<h1>💓 BP Profile Monitor</h1>"
        "<p>Sleep-aware 24-hour blood pressure review for clinicians</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    patient_details, valid = sidebar_inputs()
    profile = build_patient_profile(valid)

    doctor_tab, patient_tab, ask_tab, export_tab = st.tabs(
        ["👨‍⚕️ Doctor Dashboard", "📋 Patient Report", "Ask About This BP Report", "📥 Export"]
    )

    with doctor_tab:
        render_doctor_dashboard(valid, profile)

    with patient_tab:
        render_patient_report(valid, profile)

    with ask_tab:
        render_report_assistant(profile)

    with export_tab:
        render_export(valid, profile, patient_details)


# ── Sidebar ────────────────────────────────────────────────────────────────

def sidebar_inputs() -> tuple[dict[str, str], pd.DataFrame]:
    st.sidebar.markdown("### 🩺 Patient Details")
    patient_id = st.sidebar.text_input("Patient ID", "EXAMPLE")
    patient_name = st.sidebar.text_input("Patient Name", "Example Patient")
    age = st.sidebar.text_input("Age", "55")
    sex = st.sidebar.selectbox("Sex", ["Not recorded", "Female", "Male", "Other"], index=1)
    bmi = st.sidebar.text_input("BMI", "28")
    abpm_date = st.sidebar.date_input("ABPM date", date.today())

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Data Source")

    source = st.sidebar.radio(
        "Choose how to load data",
        ["Example patient", "Upload ABPM file", "Enter readings manually"],
        help="Upload a CSV/Excel or type BP readings by hand.",
    )
    sleep_start = st.sidebar.time_input("Usual sleep time", value=pd.Timestamp("22:00").time())
    sleep_end = st.sidebar.time_input("Usual wake time", value=pd.Timestamp("07:00").time())

    if source == "Example patient":
        valid = example_patient_abpm()
        detected_details = {}
    elif source == "Upload ABPM file":
        uploaded = st.sidebar.file_uploader(
            "ABPM file",
            type=["csv", "xlsx", "xls"],
            help=(
                "Minimum columns: Time, Systolic, Diastolic. Optional patient columns "
                "Patient_ID, Patient_Name, Age, Sex, BMI and ABPM_Date are loaded automatically."
            ),
        )
        if uploaded is None:
            st.info("⬆️ Upload an ABPM file in the sidebar, or switch to *Example patient*.")
            valid = example_patient_abpm()
            detected_details = {}
        else:
            fallback_details = {
                "Patient ID": patient_id,
                "Patient Name": patient_name,
                "Age": age,
                "Sex": sex,
                "BMI": bmi,
                "ABPM date": str(abpm_date),
            }
            valid, detected_details = _read_uploaded_abpm(
                uploaded, fallback_details, sleep_start, sleep_end
            )
            detected_display = {key: value for key, value in detected_details.items() if value}
            if detected_display:
                st.sidebar.success("Loaded patient details from file.")
    else:
        valid = _manual_entry_sidebar(patient_id, sleep_start, sleep_end)
        detected_details = {}

    patient_details = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "ABPM date": str(abpm_date),
    }
    patient_details.update({key: value for key, value in detected_details.items() if value})
    return patient_details, valid


def _read_uploaded_abpm(
    uploaded: Any,
    fallback_details: dict[str, str],
    sleep_start: Any,
    sleep_end: Any,
) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
        detected_details = extract_patient_details(raw, fallback_details)
        patient_id = detected_details.get("Patient ID") or fallback_details.get("Patient ID") or "NEW"
        valid = prepare_patient_abpm(
            raw,
            patient_id=patient_id,
            sleep_start=sleep_start.strftime("%H:%M"),
            sleep_end=sleep_end.strftime("%H:%M"),
        )
        return valid, detected_details
    except Exception as exc:
        st.error(f"Could not analyse uploaded file: {exc}")
        st.stop()


# ── Manual Entry ──────────────────────────────────────────────────────────

def _manual_entry_sidebar(patient_id: str, sleep_start: Any, sleep_end: Any) -> pd.DataFrame:
    if "manual_readings" not in st.session_state:
        st.session_state.manual_readings = []
    if "manual_entry_message" not in st.session_state:
        st.session_state.manual_entry_message = ""

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ✏️ Add a Reading")

    with st.sidebar.form("add_reading_form", clear_on_submit=True):
        reading_time = st.time_input("Time of reading", value=dt_time(8, 0))
        col_a, col_b = st.columns(2)
        systolic = col_a.number_input("Top number (SBP)", 60, 260, 130, step=1)
        diastolic = col_b.number_input("Bottom number (DBP)", 30, 160, 80, step=1)
        heart_rate = st.number_input("Pulse (heart rate)", 30, 200, 72, step=1)
        state = st.selectbox("Was the patient…", ["Awake", "Asleep"])
        add_btn = st.form_submit_button("➕ Add Reading")
        if add_btn:
            new_reading = {
                "Time": reading_time.strftime("%H:%M:%S"),
                "Systolic": int(systolic),
                "Diastolic": int(diastolic),
                "HR": int(heart_rate),
                "Wake_Sleep": 1 if state == "Awake" else 0,
            }
            st.session_state.manual_readings.append(new_reading)
            st.session_state.manual_entry_message = (
                f"Added {new_reading['Time']} - {new_reading['Systolic']}/"
                f"{new_reading['Diastolic']} mmHg, HR {new_reading['HR']} "
                f"({'Awake' if new_reading['Wake_Sleep'] == 1 else 'Asleep'})."
            )

    readings = st.session_state.manual_readings

    if st.session_state.manual_entry_message:
        st.sidebar.markdown(
            f'<div class="manual-status">{st.session_state.manual_entry_message}</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("#### Current manual readings")
    awake_count = sum(1 for item in readings if item.get("Wake_Sleep") == 1)
    sleep_count = sum(1 for item in readings if item.get("Wake_Sleep") == 0)
    status_class = "" if len(readings) >= 6 else " warn"
    st.sidebar.markdown(
        f'<div class="manual-status{status_class}">'
        f'{len(readings)} reading(s) entered. Awake: {awake_count}. Sleep: {sleep_count}.'
        f'</div>',
        unsafe_allow_html=True,
    )

    if readings:
        last = readings[-1]
        st.sidebar.markdown(
            '<div class="manual-last">'
            '<strong>Last added:</strong><br>'
            f"{last['Time']} - {last['Systolic']}/{last['Diastolic']} mmHg, "
            f"HR {last['HR']} ({'Awake' if last['Wake_Sleep'] == 1 else 'Asleep'})"
            '</div>',
            unsafe_allow_html=True,
        )
        preview_df = _manual_readings_preview(readings)
        st.sidebar.dataframe(preview_df, hide_index=True, use_container_width=True, height=220)
    else:
        st.sidebar.caption("No manual readings added yet.")

    action_col_a, action_col_b = st.sidebar.columns(2)
    if action_col_a.button("↩️ Remove last", disabled=not readings, use_container_width=True):
        removed = st.session_state.manual_readings.pop()
        st.session_state.manual_entry_message = (
            f"Removed last reading: {removed['Time']} - "
            f"{removed['Systolic']}/{removed['Diastolic']} mmHg."
        )
        st.rerun()

    if action_col_b.button("🗑️ Clear all", disabled=not readings, use_container_width=True):
        removed_count = len(st.session_state.manual_readings)
        st.session_state.manual_readings = []
        st.session_state.manual_entry_message = f"Cleared {removed_count} manual reading(s)."
        st.rerun()

    if st.sidebar.button("📋 Load example set (18 readings)", use_container_width=True):
        st.session_state.manual_readings = _example_manual_readings()
        st.session_state.manual_entry_message = "Loaded 18 example manual readings."
        st.rerun()

    readings = st.session_state.manual_readings

    if len(readings) < 6:
        st.info(
            f"✏️ You have entered **{len(readings)}** reading(s). "
            "Add at least **6** readings (including some sleep readings) for a useful analysis."
        )
        return example_patient_abpm()

    df = pd.DataFrame(readings)
    return prepare_patient_abpm(
        df,
        patient_id=patient_id or "MANUAL",
        sleep_start=sleep_start.strftime("%H:%M"),
        sleep_end=sleep_end.strftime("%H:%M"),
    )


def _manual_readings_preview(readings: list[dict]) -> pd.DataFrame:
    preview = pd.DataFrame(readings).copy()
    preview.insert(0, "#", range(1, len(preview) + 1))
    preview["State"] = preview["Wake_Sleep"].map({1: "Awake", 0: "Asleep"})
    preview = preview.rename(
        columns={
            "Time": "Time",
            "Systolic": "SBP",
            "Diastolic": "DBP",
            "HR": "HR",
        }
    )
    return preview[["#", "Time", "SBP", "DBP", "HR", "State"]]


def _example_manual_readings() -> list[dict]:
    """Pre-built set of 18 readings matching the example patient."""
    hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 23]
    sbp = [139, 136, 138, 140, 137, 139, 136, 160, 164, 151, 136, 133, 140, 135, 142, 134, 139, 137]
    dbp = [84, 82, 83, 85, 82, 84, 83, 96, 99, 91, 84, 80, 85, 82, 86, 81, 84, 83]
    hr = [72, 70, 69, 70, 71, 70, 72, 82, 86, 80, 75, 73, 74, 72, 76, 73, 72, 71]
    ws = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    return [
        {"Time": f"{h:02d}:00:00", "Systolic": s, "Diastolic": d, "HR": r, "Wake_Sleep": w}
        for h, s, d, r, w in zip(hours, sbp, dbp, hr, ws)
    ]


# ── Doctor Dashboard ──────────────────────────────────────────────────────

def render_doctor_dashboard(valid: pd.DataFrame, profile: dict[str, Any]) -> None:
    # ── 1. Traffic-light summary banner ──
    level = priority_level(profile.get("priority", ""))
    label = profile_label(profile)
    caption = _curve_caption(profile)
    emoji_map = {"green": "✅", "yellow": "⚠️", "red": "🔴", "grey": "⚪"}
    st.markdown(
        f'<div class="traffic-banner">'
        f'<div class="traffic-stripe {level}"></div>'
        f'<div class="traffic-body">'
        f'<span class="tb-profile">Overall BP Profile: {label}</span>'
        f'<span class="tb-priority {level}">{emoji_map.get(level, "")} {profile.get("priority", "")}</span>'
        f'<div class="tb-issue"><strong>Main finding:</strong> {caption}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── 2. Summary cards ──
    _render_summary_cards(profile)

    # ── 3. 24h BP curve with caption ──
    st.markdown("#### 📈 24-Hour Blood Pressure Curve")
    fig = plot_24h_bp_curve(valid, profile)
    st.pyplot(fig, clear_figure=True)
    is_ok = caption.startswith("No major")
    st.markdown(
        f'<div class="curve-caption {"ok" if is_ok else ""}">📋 {caption}</div>',
        unsafe_allow_html=True,
    )

    # ── 4. Day-vs-night dipping visual + Awake/Sleep bar ──
    dip_col, bar_col = st.columns([1.05, 0.95])

    with dip_col:
        st.markdown("#### ☀️🌙 Day vs Night BP")
        awake_sbp = profile.get("awake_mean_sbp")
        sleep_sbp = profile.get("sleep_mean_sbp")
        dip_pct = profile.get("dipping_pct_sbp")
        awake_str = f"{awake_sbp:.0f}" if pd.notna(awake_sbp) else "—"
        sleep_str = f"{sleep_sbp:.0f}" if pd.notna(sleep_sbp) else "—"
        is_non_dip = profile.get("dipping_category") in ("non_dipper", "reverse_dipper")

        if dip_pct is not None and not pd.isna(dip_pct):
            dip_text = f"{abs(dip_pct):.1f}% {'fall' if dip_pct > 0 else 'rise'}"
        else:
            dip_text = "N/A"

        st.markdown(
            f'<div class="dip-flow">'
            f'<div class="dip-box"><div class="label">☀️ Awake SBP</div><div class="val">{awake_str}</div></div>'
            f'<span class="dip-arrow">→</span>'
            f'<div class="dip-box"><div class="label">🌙 Sleep SBP</div><div class="val">{sleep_str}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        expected_text = "Expected: should fall by about 10–20%"
        observed_class = 'style="font-weight:600; color:#a16207;"' if is_non_dip else 'style="font-weight:600;"'
        st.markdown(
            f'<div style="border-left:3px solid #cbd5e1; padding:0.4rem 0.8rem; font-size:0.85rem; color:#334155;">'
            f'<div style="font-weight:500;">{expected_text}</div>'
            f'<div {observed_class}>Observed: {dip_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with bar_col:
        st.markdown("#### Awake vs Sleep BP")
        st.pyplot(plot_awake_sleep_bar(profile), clear_figure=True)

    # ── 5. Profile position with explanation ──
    pos_col, explain_col = st.columns([1.35, 0.65], gap="small")
    with pos_col:
        st.markdown("#### 📍 Profile Position")
        st.pyplot(plot_profile_position(profile), clear_figure=True)
    with explain_col:
        st.markdown("#### Reading the chart")
        st.markdown(
            '<div class="reading-card">'
            '<p><strong>Left side:</strong> BP did not fall enough during sleep.</p>'
            '<p><strong>Higher position:</strong> stronger BP rise after waking.</p>'
            f'<p><strong>This patient:</strong> {label}</p>'
            '<p class="axis-note">X-axis = sleep BP drop. Y-axis = morning BP rise.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── 6. Clinical status table (replaces pattern flags) ──
    st.markdown("#### 🩺 Pattern Status")
    status_rows = _build_clinical_status_table(profile)
    status_df = pd.DataFrame(status_rows)
    status_df.columns = ["Pattern", "Status", "Why it matters", "Review point"]
    st.dataframe(status_df, hide_index=True, use_container_width=True)

    # ── 7. Patient-friendly summary card ──
    explanation = profile.get("patient_explanation", "")
    reviews = review_points(profile)
    review_tags = " ".join(
        f'<span style="background:#e0e7ff;color:#3730a3;padding:0.15rem 0.55rem;'
        f'border-radius:14px;font-size:0.78rem;font-weight:600;margin-right:0.3rem;">'
        f'{r["Doctor review point"]}</span>'
        for r in reviews
    )
    st.markdown(
        f'<div class="patient-summary-card">'
        f'<h3>💬 Your main BP pattern</h3>'
        f'<div class="summary-text">{explanation}</div>'
        f'<div style="margin-top:0.7rem;">{review_tags}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 8. Data quality box ──
    st.markdown("#### 📋 Data Quality")
    dq_cols = st.columns(4)
    dq_cols[0].metric("Valid readings", profile.get("valid_readings", "—"))
    dq_cols[1].metric("Sleep readings", profile.get("sleep_valid_readings", "—"))
    dq_cols[2].metric("Sleep/wake source", "Entered sleep time")
    dq_cols[3].metric("Quality", profile.get("data_quality", "—"))

    # ── 9. Data preview ──
    st.markdown("#### 📊 Patient Readings Preview")
    preview = valid[["measurement_datetime", "Systolic", "Diastolic", "HR", "Wake_Sleep"]].head(8).copy()
    preview.columns = ["Time", "Top (SBP)", "Bottom (DBP)", "Pulse", "Awake/Asleep"]
    preview["Awake/Asleep"] = preview["Awake/Asleep"].map({1: "☀️ Awake", 0: "🌙 Asleep"})
    preview["Time"] = preview["Time"].dt.strftime("%d %b %Y  %H:%M")
    st.dataframe(preview, hide_index=True, use_container_width=True)

    st.warning(
        "⚠️ This dashboard supports clinician review. "
        "It does **not** recommend automatic medication changes."
    )



# ── Summary cards — uses st.columns to avoid raw-HTML rendering bug ──

_LEVEL_COLOURS = {
    "green": "#22c55e",
    "yellow": "#eab308",
    "red": "#ef4444",
    "grey": "#94a3b8",
}


def _card_html(label: str, value: str, detail: str, level: str) -> str:
    colour = _LEVEL_COLOURS.get(level, "#94a3b8")
    return (
        f'<div style="border:1px solid #e2e8f0; border-left:5px solid {colour};'
        f' border-radius:10px; padding:0.9rem 1rem; background:#ffffff;'
        f' min-height:110px; box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
        f'<p style="color:#334155; font-size:0.72rem; font-weight:750;'
        f' text-transform:uppercase; letter-spacing:0.04em; margin:0;">{label}</p>'
        f'<p style="color:#020617; font-size:1.15rem; font-weight:800;'
        f' margin:0.3rem 0 0.1rem; word-break:break-word;">{value}</p>'
        f'<p style="color:#475569; font-size:0.82rem; font-weight:500; margin:0;">{detail}</p>'
        f"</div>"
    )


def _render_summary_cards(profile: dict[str, Any]) -> None:
    cards = [
        {
            "label": "24h Average BP",
            "value": _bp_value(profile.get("mean_24h_sbp"), profile.get("mean_24h_dbp")),
            "detail": "Full-day ABPM average",
            "level": "yellow" if profile.get("hypertensive_24h") else "green",
        },
        {
            "label": "Awake BP",
            "value": _bp_value(profile.get("awake_mean_sbp"), profile.get("awake_mean_dbp")),
            "detail": "Daytime readings",
            "level": "yellow" if profile.get("hypertensive_awake") else "green",
        },
        {
            "label": "Sleep BP",
            "value": _bp_value(profile.get("sleep_mean_sbp"), profile.get("sleep_mean_dbp")),
            "detail": "Night-time readings",
            "level": _sleep_level(profile),
        },
        {
            "label": "Dipping Status",
            "value": _pretty_category(profile.get("dipping_category")),
            "detail": "How much BP drops at night",
            "level": _dipping_level(profile),
        },
        {
            "label": "Morning Surge",
            "value": f"{profile['morning_surge_sbp']:.0f} mmHg"
            if pd.notna(profile.get("morning_surge_sbp"))
            else "N/A",
            "detail": "BP rise in the first 2 h after waking",
            "level": _morning_surge_level(profile),
        },
        {
            "label": "BP Variability",
            "value": "High" if profile.get("high_variability") else "Normal",
            "detail": f"SBP spread: {profile['sbp_sd']:.1f} mmHg",
            "level": "yellow" if profile.get("high_variability") else "green",
        },
        {
            "label": "Priority",
            "value": profile.get("priority", ""),
            "detail": profile.get("data_quality", ""),
            "level": priority_level(profile.get("priority", "")),
        },
        {
            "label": "Valid Readings",
            "value": str(profile.get("valid_readings", "")),
            "detail": f"Sleep readings: {profile.get('sleep_valid_readings', '')}",
            "level": "grey" if profile.get("sleep_valid_readings", 0) < 3 else "green",
        },
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)
    for idx, card in enumerate(cards):
        col = row1[idx] if idx < 4 else row2[idx - 4]
        with col:
            st.markdown(
                _card_html(card["label"], card["value"], card["detail"], card["level"]),
                unsafe_allow_html=True,
            )


# ── Patient report ────────────────────────────────────────────────────────

def render_patient_report(valid: pd.DataFrame, profile: dict[str, Any]) -> None:
    st.subheader("Your 24-Hour Blood Pressure Pattern")
    st.markdown(
        f"<div class='patient-note'>{profile['patient_explanation']}</div>",
        unsafe_allow_html=True,
    )

    st.pyplot(plot_24h_bp_curve(valid, profile, patient_view=True), clear_figure=True)
    st.markdown("#### What This Means")
    st.write(_patient_plain_summary(profile))

    st.markdown("#### Safe Next Steps")
    st.markdown(
        """
<div class="safety-box">
✅ Continue your prescribed medicine.<br>
🚫 Do not change your dose without your doctor.<br>
📏 Measure BP as advised.<br>
📄 Bring this report to your next appointment.<br>
📞 Contact your clinic if readings stay high.<br>
🚨 Seek urgent help if high BP comes with chest pain, weakness, severe headache,
confusion or shortness of breath.
</div>
""",
        unsafe_allow_html=True,
    )


# ── Report assistant ───────────────────────────────────────────────────────

def render_report_assistant(profile: dict[str, Any]) -> None:
    st.subheader("Ask About This BP Report")
    st.caption(
        "This explains the already calculated BP report. It does not inspect raw readings, diagnose, or recommend medication changes."
    )

    context = build_report_context(profile)
    with st.expander("Report summary sent to the assistant", expanded=False):
        st.json(context)

    if "assistant_transcript" not in st.session_state:
        st.session_state.assistant_transcript = []
    if "assistant_question" not in st.session_state:
        st.session_state.assistant_question = ""

    st.markdown("#### Quick questions")
    q_cols = st.columns(len(quick_questions()))
    for col, (label, question) in zip(q_cols, quick_questions().items()):
        if col.button(label, use_container_width=True, key=f"assistant_quick_{label}"):
            st.session_state.assistant_question = question
            st.session_state.assistant_pending_question = question

    st.markdown("#### Custom question")
    st.text_area(
        "Ask a question about the current report",
        key="assistant_question",
        placeholder="Example: Why is this patient flagged?",
        height=90,
    )

    hf_status = token_status()["Hugging Face Gemma 4"]
    st.info(
        f"Assistant model: Hugging Face Gemma 4. Token status: {hf_status}. "
        "Only the calculated report summary is sent to the model."
    )

    ask_clicked = st.button("Ask about this report", type="primary")
    if ask_clicked:
        st.session_state.assistant_pending_question = st.session_state.assistant_question

    pending_question = st.session_state.pop("assistant_pending_question", None)
    if pending_question is not None:
        question_to_ask = str(pending_question).strip()
        if not question_to_ask:
            st.warning("Type a question first, or choose one of the quick questions.")
            return
        with st.spinner("Preparing report explanation..."):
            try:
                response = answer_report_question(
                    question_to_ask,
                    context,
                    model=DEFAULT_HF_MODEL,
                )
                st.session_state.assistant_transcript.append(
                    {
                        "question": question_to_ask,
                        "answer": response.answer,
                        "source": response.source,
                    }
                )
            except Exception as exc:
                st.error(f"Assistant request failed: {exc}")

    if st.session_state.assistant_transcript:
        st.markdown("#### Answers")
        for item in reversed(st.session_state.assistant_transcript[-6:]):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer source:** {item['source']}")
            st.markdown(item["answer"])
            st.divider()

        transcript_text = _assistant_transcript_text()
        st.download_button(
            "Download assistant Q&A",
            data=transcript_text.encode("utf-8"),
            file_name="bp_report_assistant_summary.txt",
            mime="text/plain",
        )


# ── Export ─────────────────────────────────────────────────────────────────

def render_export(valid: pd.DataFrame, profile: dict[str, Any], patient_details: dict[str, str]) -> None:
    st.subheader("Export Report")
    st.write("The PDF keeps the same rule-based clinical interpretation and patient safety wording.")

    assistant_summary = _assistant_transcript_text() if st.session_state.get("assistant_transcript") else None
    pdf_bytes = create_pdf_report(valid, profile, patient_details, assistant_summary=assistant_summary)
    st.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"bp_report_{patient_details['Patient ID']}.pdf",
        mime="application/pdf",
    )

    table = feature_table(profile)
    csv_buffer = BytesIO()
    table.to_csv(csv_buffer, index=False)
    st.download_button(
        "📥 Download Feature Table CSV",
        data=csv_buffer.getvalue(),
        file_name=f"bp_features_{patient_details['Patient ID']}.csv",
        mime="text/csv",
    )

    st.markdown("#### Feature Table Preview")
    st.dataframe(table, hide_index=True, use_container_width=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def _bp_value(sbp: Any, dbp: Any) -> str:
    if pd.isna(sbp) or pd.isna(dbp):
        return "N/A"
    return f"{float(sbp):.0f}/{float(dbp):.0f} mmHg"


def _pretty_category(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(value).replace("_", " ").title()


def _sleep_level(profile: dict[str, Any]) -> str:
    if profile.get("sleep_valid_readings", 0) < 3:
        return "grey"
    return "yellow" if _is_flagged(profile.get("hypertensive_sleep")) else "green"


def _morning_surge_level(profile: dict[str, Any]) -> str:
    if pd.isna(profile.get("morning_surge_sbp")):
        return "grey"
    return "yellow" if profile.get("morning_surge_high") else "green"


def _dipping_level(profile: dict[str, Any]) -> str:
    category = profile.get("dipping_category")
    if category == "insufficient_sleep":
        return "grey"
    if category == "reverse_dipper":
        return "red"
    if category in {"non_dipper", "extreme_dipper"}:
        return "yellow"
    return "green"


def _curve_caption(profile: dict[str, Any]) -> str:
    clauses = []
    if _is_flagged(profile.get("hypertensive_sleep")):
        clauses.append("BP stayed high during sleep")
    if profile.get("morning_surge_high"):
        clauses.append("rose further after waking")
    if profile.get("high_variability"):
        clauses.append("showed high variability")
    if not clauses:
        return "No major BP pattern warning was found in this recording."
    return "BP " + ", and ".join(clauses) + "."


def _patient_plain_summary(profile: dict[str, Any]) -> str:
    if profile.get("dipping_category") == "insufficient_sleep":
        return "There were not enough sleep readings to describe the night-time pattern reliably."
    messages = []
    if profile.get("dipping_category") in {"non_dipper", "reverse_dipper"}:
        messages.append("BP did not fall enough during sleep")
    if profile.get("morning_surge_high"):
        messages.append("BP increased after waking")
    if profile.get("high_variability"):
        messages.append("readings changed more than expected")
    if _is_flagged(profile.get("sustained_high_bp")):
        messages.append("BP was high across the 24-hour recording")
    if not messages:
        return "The recording did not show a major warning flag."
    return (
        "The main finding is that "
        + ", and ".join(messages)
        + ". Your doctor may use this to review night BP, sleep quality, "
        "morning BP control, stress or caffeine triggers and medication timing."
    )


def _is_flagged(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


def _assistant_transcript_text() -> str:
    rows = st.session_state.get("assistant_transcript", [])
    if not rows:
        return ""
    parts = [
        "Ask About This BP Report",
        "This summary explains the calculated report only. It is not a diagnosis or medication instruction.",
    ]
    for idx, item in enumerate(rows, start=1):
        parts.append(
            f"\nQ{idx}: {item['question']}\n"
            f"Source: {item['source']}\n"
            f"A{idx}: {item['answer']}"
        )
    return "\n".join(parts)

def _build_clinical_status_table(profile: dict[str, Any]) -> list[dict]:
    """Build doctor-friendly combined status table."""
    rows = []
    dipping = profile.get("dipping_category")

    if dipping in ("non_dipper", "reverse_dipper"):
        rows.append({"p": "Sleep BP fall", "s": "Needs review",
                      "w": "BP did not fall enough during sleep",
                      "r": "Review night BP and sleep quality"})
    else:
        rows.append({"p": "Sleep BP fall", "s": "Normal",
                      "w": "BP dipped within expected range during sleep",
                      "r": "Routine review"})

    if profile.get("morning_surge_high"):
        rows.append({"p": "Morning rise", "s": "Needs review",
                      "w": "BP increased after waking",
                      "r": "Review morning BP control"})
    else:
        rows.append({"p": "Morning rise", "s": "Normal",
                      "w": "Morning BP rise within expected range",
                      "r": "Routine review"})

    if _is_flagged(profile.get("sustained_high_bp")):
        rows.append({"p": "BP burden", "s": "Needs review",
                      "w": "BP stayed high across monitoring",
                      "r": "Consider earlier treatment review"})
    else:
        rows.append({"p": "BP burden", "s": "Normal",
                      "w": "24h BP average within expected range",
                      "r": "Routine review"})

    pp = (profile.get("mean_24h_sbp") or 0) - (profile.get("mean_24h_dbp") or 0)
    if pp and pp > 60:
        rows.append({"p": "Pressure gap", "s": "Needs review",
                      "w": "Wide gap between top and bottom numbers",
                      "r": "Review arterial stiffness indicators"})
    else:
        rows.append({"p": "Pressure gap", "s": "Within range",
                      "w": "Gap between top and bottom numbers is normal",
                      "r": "Routine review"})

    if profile.get("high_variability"):
        rows.append({"p": "BP variability", "s": "Needs review",
                      "w": "Readings changed more than expected",
                      "r": "Check stress, caffeine, adherence"})
    else:
        rows.append({"p": "BP variability", "s": "Normal",
                      "w": "Readings were within expected variation",
                      "r": "Routine review"})

    return rows


if __name__ == "__main__":
    main()
