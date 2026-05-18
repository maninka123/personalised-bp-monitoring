# Sample Patient Inputs

These files show the ABPM upload format used by the Streamlit dashboard and the desktop app.

Each row is one blood pressure reading. Patient details are repeated on each row so one CSV contains everything needed for a new patient report.

Required columns:

| Column | Meaning |
|---|---|
| Patient_ID | Patient or study identifier |
| Patient_Name | Patient display name |
| Age | Age in years |
| Sex | Female, Male, Other, or Not recorded |
| BMI | Body mass index |
| ABPM_Date | Recording date |
| Day_Date | Date for the reading |
| Time | Reading time, for example 07:00 |
| Systolic | Systolic BP in mmHg |
| Diastolic | Diastolic BP in mmHg |
| HR | Heart rate |
| Wake_Sleep | 1 = awake, 0 = asleep |

Optional columns:

| Column | Meaning |
|---|---|
| MAP | Mean arterial pressure. If missing, the app calculates it. |
| PP | Pulse pressure. If missing, the app calculates it. |

Use `TEMPLATE_ABPM_with_patient_details.csv` when entering a real or simulated patient manually.
The 10 sample files are fictional examples for testing the interface only.
