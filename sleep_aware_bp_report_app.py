from __future__ import annotations

from datetime import date
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from clinical_report_utils import (
    build_patient_profile,
    create_pdf_report,
    example_patient_abpm,
    feature_table,
    pattern_flags,
    plot_24h_bp_curve,
    plot_awake_sleep_bar,
    plot_profile_position,
    prepare_patient_abpm,
    priority_level,
    review_points,
)


st.set_page_config(
    page_title="Sleep-Aware BP Report",
    layout="wide",
)


CARD_CSS = """
<style>
.bp-card-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
}
.bp-card {
    border: 1px solid #d8dee8;
    border-left-width: 6px;
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    background: #ffffff;
    min-height: 104px;
}
.bp-card .label {
    color: #5e6b78;
    font-size: 0.78rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0;
}
.bp-card .value {
    color: #17212b;
    font-size: 1.4rem;
    font-weight: 750;
    margin-top: 0.3rem;
}
.bp-card .detail {
    color: #5b6872;
    font-size: 0.86rem;
    margin-top: 0.15rem;
}
.green { border-left-color: #2f8f5b; }
.yellow { border-left-color: #c99325; }
.red { border-left-color: #c84b3f; }
.grey { border-left-color: #7c8792; }
.patient-note {
    border-left: 6px solid #2f5d7c;
    background: #f4f8fb;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    font-size: 1.04rem;
    line-height: 1.55;
}
.safety-box {
    border: 1px solid #d8dee8;
    background: #fffdf7;
    border-radius: 8px;
    padding: 1rem 1.1rem;
}
@media (max-width: 1100px) {
    .bp-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


def main() -> None:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("Sleep-Aware BP Profile Report")
    st.caption(
        "Rule-based 24-hour ABPM review for clinicians, with a patient-friendly report preview."
    )

    patient_details, valid = sidebar_inputs()
    profile = build_patient_profile(valid)

    doctor_tab, patient_tab, export_tab = st.tabs(
        ["Doctor dashboard", "Patient report preview", "Export report"]
    )

    with doctor_tab:
        render_doctor_dashboard(valid, profile)

    with patient_tab:
        render_patient_report(valid, profile)

    with export_tab:
        render_export(valid, profile, patient_details)


def sidebar_inputs() -> tuple[dict[str, str], pd.DataFrame]:
    st.sidebar.header("New patient")
    patient_id = st.sidebar.text_input("Patient ID", "EXAMPLE")
    age = st.sidebar.text_input("Age", "55")
    sex = st.sidebar.selectbox("Sex", ["Not recorded", "Female", "Male", "Other"], index=1)
    bmi = st.sidebar.text_input("BMI", "28")
    abpm_date = st.sidebar.date_input("ABPM date", date.today())

    source = st.sidebar.radio(
        "Data source",
        ["Example patient", "Upload ABPM file"],
        help="Upload CSV or Excel with Time, Systolic and Diastolic. Wake_Sleep is optional.",
    )
    sleep_start = st.sidebar.time_input("Usual sleep start", value=pd.Timestamp("22:00").time())
    sleep_end = st.sidebar.time_input("Usual wake time", value=pd.Timestamp("07:00").time())

    if source == "Example patient":
        valid = example_patient_abpm()
    else:
        uploaded = st.sidebar.file_uploader("ABPM file", type=["csv", "xlsx", "xls"])
        if uploaded is None:
            st.info("Upload an ABPM file or use the example patient to view the dashboard.")
            valid = example_patient_abpm()
        else:
            valid = read_uploaded_abpm(uploaded, patient_id, sleep_start, sleep_end)

    patient_details = {
        "Patient ID": patient_id,
        "Age": age,
        "Sex": sex,
        "BMI": bmi,
        "ABPM date": str(abpm_date),
    }
    return patient_details, valid


def read_uploaded_abpm(uploaded: Any, patient_id: str, sleep_start: Any, sleep_end: Any) -> pd.DataFrame:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)
        return prepare_patient_abpm(
            raw,
            patient_id=patient_id or "NEW",
            sleep_start=sleep_start.strftime("%H:%M"),
            sleep_end=sleep_end.strftime("%H:%M"),
        )
    except Exception as exc:
        st.error(f"Could not analyse uploaded file: {exc}")
        st.stop()


def render_doctor_dashboard(valid: pd.DataFrame, profile: dict[str, Any]) -> None:
    st.subheader("Doctor dashboard")
    render_summary_cards(profile)

    st.markdown("#### 24-hour BP curve")
    fig = plot_24h_bp_curve(valid, profile)
    st.pyplot(fig, clear_figure=True)
    st.caption(curve_caption(profile))

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("#### Profile position plot")
        st.pyplot(plot_profile_position(profile), clear_figure=True)
        st.caption("X-axis shows sleep BP fall. Y-axis shows the BP rise after waking.")

    with right:
        st.markdown("#### Awake vs sleep BP")
        st.pyplot(plot_awake_sleep_bar(profile), clear_figure=True)

    flags_col, review_col = st.columns([0.9, 1.1])
    with flags_col:
        st.markdown("#### Pattern flags")
        flags = pd.DataFrame(pattern_flags(profile))
        st.dataframe(flags, hide_index=True, use_container_width=True)

    with review_col:
        st.markdown("#### Review points")
        points = pd.DataFrame(review_points(profile))
        st.dataframe(points, hide_index=True, use_container_width=True)

    st.warning(
        "This dashboard supports clinician review. It does not recommend automatic medication changes."
    )


def render_summary_cards(profile: dict[str, Any]) -> None:
    cards = [
        {
            "label": "24h average BP",
            "value": bp_value(profile.get("mean_24h_sbp"), profile.get("mean_24h_dbp")),
            "detail": "ABPM average",
            "level": "yellow" if profile.get("hypertensive_24h") else "green",
        },
        {
            "label": "Awake BP",
            "value": bp_value(profile.get("awake_mean_sbp"), profile.get("awake_mean_dbp")),
            "detail": "Day or awake readings",
            "level": "yellow" if profile.get("hypertensive_awake") else "green",
        },
        {
            "label": "Sleep BP",
            "value": bp_value(profile.get("sleep_mean_sbp"), profile.get("sleep_mean_dbp")),
            "detail": "Night or sleep readings",
            "level": sleep_level(profile),
        },
        {
            "label": "Dipping status",
            "value": pretty_category(profile.get("dipping_category")),
            "detail": "Sleep BP fall",
            "level": dipping_level(profile),
        },
        {
            "label": "Morning surge",
            "value": f"{profile['morning_surge_sbp']:.0f} mmHg"
            if pd.notna(profile.get("morning_surge_sbp"))
            else "N/A",
            "detail": "First 2 hours after waking",
            "level": morning_surge_level(profile),
        },
        {
            "label": "BP variability",
            "value": "High" if profile.get("high_variability") else "Not flagged",
            "detail": f"SBP SD {profile['sbp_sd']:.1f} mmHg",
            "level": "yellow" if profile.get("high_variability") else "green",
        },
        {
            "label": "Priority",
            "value": profile.get("priority", ""),
            "detail": profile.get("data_quality", ""),
            "level": priority_level(profile.get("priority", "")),
        },
        {
            "label": "Valid readings",
            "value": str(profile.get("valid_readings", "")),
            "detail": f"Sleep readings: {profile.get('sleep_valid_readings', '')}",
            "level": "grey" if profile.get("sleep_valid_readings", 0) < 3 else "green",
        },
    ]
    html = ['<div class="bp-card-grid">']
    for card in cards:
        html.append(
            f"""
            <div class="bp-card {card['level']}">
                <div class="label">{card['label']}</div>
                <div class="value">{card['value']}</div>
                <div class="detail">{card['detail']}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_patient_report(valid: pd.DataFrame, profile: dict[str, Any]) -> None:
    st.subheader("Your 24-hour Blood Pressure Pattern")
    st.markdown(f"<div class='patient-note'>{profile['patient_explanation']}</div>", unsafe_allow_html=True)

    st.pyplot(plot_24h_bp_curve(valid, profile, patient_view=True), clear_figure=True)
    st.markdown("#### What this means")
    st.write(patient_plain_summary(profile))

    st.markdown("#### Safe next steps")
    st.markdown(
        """
<div class="safety-box">
Please continue your prescribed medicine.<br>
Do not change your dose without your doctor.<br>
Measure BP as advised.<br>
Bring this report to your next appointment.<br>
Contact your clinic if readings stay high.<br>
Seek urgent help if high BP is associated with chest pain, weakness, severe headache,
confusion or shortness of breath.
</div>
""",
        unsafe_allow_html=True,
    )


def render_export(valid: pd.DataFrame, profile: dict[str, Any], patient_details: dict[str, str]) -> None:
    st.subheader("Export report")
    st.write("The PDF keeps the same rule-based clinical interpretation and patient safety wording.")

    pdf_bytes = create_pdf_report(valid, profile, patient_details)
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"sleep_aware_bp_report_{patient_details['Patient ID']}.pdf",
        mime="application/pdf",
    )

    table = feature_table(profile)
    csv_buffer = BytesIO()
    table.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download feature table CSV",
        data=csv_buffer.getvalue(),
        file_name=f"sleep_aware_bp_features_{patient_details['Patient ID']}.csv",
        mime="text/csv",
    )

    st.markdown("#### Feature table preview")
    st.dataframe(table, hide_index=True, use_container_width=True)


def bp_value(sbp: Any, dbp: Any) -> str:
    if pd.isna(sbp) or pd.isna(dbp):
        return "N/A"
    return f"{float(sbp):.0f}/{float(dbp):.0f} mmHg"


def pretty_category(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return str(value).replace("_", " ").title()


def sleep_level(profile: dict[str, Any]) -> str:
    if profile.get("sleep_valid_readings", 0) < 3:
        return "grey"
    return "yellow" if is_flagged(profile.get("hypertensive_sleep")) else "green"


def morning_surge_level(profile: dict[str, Any]) -> str:
    if pd.isna(profile.get("morning_surge_sbp")):
        return "grey"
    return "yellow" if profile.get("morning_surge_high") else "green"


def dipping_level(profile: dict[str, Any]) -> str:
    category = profile.get("dipping_category")
    if category == "insufficient_sleep":
        return "grey"
    if category == "reverse_dipper":
        return "red"
    if category in {"non_dipper", "extreme_dipper"}:
        return "yellow"
    return "green"


def curve_caption(profile: dict[str, Any]) -> str:
    clauses = []
    if is_flagged(profile.get("hypertensive_sleep")):
        clauses.append("BP stayed high during sleep")
    if profile.get("morning_surge_high"):
        clauses.append("rose further after waking")
    if profile.get("high_variability"):
        clauses.append("showed high variability")
    if not clauses:
        return "No major rule-based BP pattern warning was detected in this profile."
    return "BP " + ", and ".join(clauses) + "."


def patient_plain_summary(profile: dict[str, Any]) -> str:
    if profile.get("dipping_category") == "insufficient_sleep":
        return "There were not enough sleep readings to describe the night-time pattern reliably."
    messages = []
    if profile.get("dipping_category") in {"non_dipper", "reverse_dipper"}:
        messages.append("BP did not fall enough during sleep")
    if profile.get("morning_surge_high"):
        messages.append("BP increased after waking")
    if profile.get("high_variability"):
        messages.append("readings changed more than expected")
    if is_flagged(profile.get("sustained_high_bp")):
        messages.append("BP was high across the 24-hour recording")
    if not messages:
        return "The profile did not show a major rule-based warning flag."
    return (
        "The main finding is that " + ", and ".join(messages) + ". "
        "Your doctor may use this to review night BP, sleep quality, morning BP control, "
        "stress or caffeine triggers and medication timing where clinically appropriate."
    )


def is_flagged(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


if __name__ == "__main__":
    main()
