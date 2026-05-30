# Sleep-Aware BP Profiling and ABPM-TSL

![Status: Research](https://img.shields.io/badge/Status-Research-blue)
![Stage: Clinical Prototype](https://img.shields.io/badge/Stage-Clinical_Prototype-brightgreen)
![Model: ABPM-TSL](https://img.shields.io/badge/Model-ABPM--TSL-orange)
![Safety: Clinician Led](https://img.shields.io/badge/Safety-Clinician_Led-red)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blueviolet)

**Tags:** hypertension monitoring, ambulatory blood pressure monitoring, ABPM, nocturnal hypertension, non-dipping, morning surge, blood pressure variability, limited-input risk estimation, teacher-student learning, missingness-aware neural network, Gemma, clinical decision support, digital health, Sri Lanka

This project helps turn blood pressure monitoring data into a clearer report for clinical review.

It has two main parts:

1. **Full ABPM report:** uses a 24-hour ABPM file to calculate sleep BP, awake BP, dipping pattern, morning surge, BP variability and review points.
2. **Limited-input risk estimate:** uses a trained ABPM-TSL neural student model to estimate ABPM-related risk patterns from clinic, home, demographic and lifestyle inputs when full ABPM is not immediately available.

The project is for research and monitoring support. It does **not** replace ABPM, diagnose hypertension by itself, or recommend medication changes.

## Full Pipeline At A Glance

![Personalised blood pressure monitoring pathway](paper/figures/overall%20image.png)

<p align="center">Overall pathway</p>

```text
Patient BP information
  -> full 24-hour ABPM when available
  -> sleep-aware BP profile
  -> clear personalised report
  -> clinician-led review

When full ABPM is not yet available
  -> clinic BP, home BP and patient context
  -> missing information is accepted
  -> ABPM-TSL neural risk model
  -> estimated ABPM risks for prioritising review
```

## Why This Matters

Clinic BP alone may miss important patterns:

- BP may be high during sleep.
- BP may not fall normally overnight.
- BP may rise sharply in the morning.
- BP may vary a lot across the day.
- A patient may need ABPM review even if limited clinic readings look acceptable.

The aim is to support clinical review and patient communication.

## What The App Does

For a full ABPM upload, the app can:

- clean invalid readings
- separate awake and sleep readings
- calculate 24-hour, awake and sleep BP
- classify dipping pattern
- flag morning surge, high variability and raised sleep BP
- create a clinician-facing report
- generate a PDF report
- explain the report summary using Gemma

![24-hour BP report example](docs/figures/new_patient_framework_example.png)

<p align="center">24-hour BP report example</p>

For limited-input use, the app can:

- accept clinic BP, home BP and patient context
- show which inputs are missing
- estimate ABPM-related risk probabilities
- suggest whether ABPM review should be routine, soon, or high priority
- clearly state that the result is prioritisation support, not a diagnosis

## ABPM-TSL Model

ABPM-TSL means **ABPM Teacher-Student Learning**.

Model structure:

```text
Teacher model:
learns from full 24-hour ABPM curves during development

Student model:
uses routinely collected inputs such as clinic BP, home BP, age, BMI, sleep quality and medical history

Output:
supports ABPM prioritisation when full ABPM is not yet available
```

The student model estimates five ABPM-related risk patterns:

- abnormal dipping
- high morning surge
- nocturnal hypertension
- high BP burden
- high BP variability

The model is **missingness-aware**, meaning it is designed to handle incomplete forms. For example, it can still run if home BP or lifestyle details are missing, but it lowers confidence and reports what information is absent.

## Current Model Status

The current neural model is trained on a synthetic limited-input cohort derived from the available ABPM dataset. It is useful for method development and app testing, but it is **not yet clinical validation**.

Key current result:

| Model | Mean AUROC | Main interpretation |
|---|---:|---|
| Full ABPM-TSL | 0.730 | Best current limited-input neural model in the synthetic development experiment |

The strongest targets were high BP burden and nocturnal hypertension. Morning surge was harder to estimate, which is expected because it is a time-specific pattern and limited inputs are less precise than full ABPM.

Full result tables are kept in:

```text
ABPM-TSL/results/
```

The trained student model used by the desktop app is:

```text
ABPM-TSL/models/student_abpm_tsl_torchscript.pt
```

Model pathway:

```text
Full ABPM data
  -> rule-derived ABPM labels
  -> teacher model learns ABPM patterns
  -> limited-input student model learns to estimate those patterns
  -> desktop app uses the trained student model when available
```

## Gemma Report Assistant ✨

The **Ask About This BP Report** assistant uses Gemma to explain the calculated report summary.

It is designed to be safe:

- it receives the report summary, not raw ABPM rows
- it explains what the report says
- it does not diagnose
- it does not prescribe
- it does not tell a patient to change medication

Example:

```bash
python ask_bp_report.py --save-token
python ask_bp_report.py --question "Why is this patient flagged?"
```

## Desktop App

The desktop app source is in:

```text
desktop_app/
```

It includes:

- ABPM upload and report dashboard
- Limited-Input Risk tab
- FastAPI backend
- trained TorchScript student model loading
- sample patient inputs

Build command:

```bash
cd desktop_app
npm install
npm run dist-win
```

The installer is unsigned, so Windows may show an unknown-publisher warning.

## Streamlit App

Run the Streamlit version:

```bash
pip install -r requirements.txt
streamlit run sleep_aware_bp_report_app.py
```

Uploaded files should include at least:

```text
Time, Systolic, Diastolic
```

Optional useful columns:

```text
Patient_ID, Patient_Name, Age, Sex, BMI, ABPM_Date,
Day_Date, MAP, PP, HR, Wake_Sleep
```

Sample fictional files are in:

```text
Sample Patient Inputs/
desktop_app/sample-patient-inputs/
```

## Project Folders

```text
ABPM-TSL/          neural teacher-student model, results, figures and model files
desktop_app/       Electron desktop app and FastAPI backend
paper/             manuscript files, figures and tables
outputs/           generated ABPM analysis outputs
Sample Patient Inputs/  fictional testing files
```

Raw datasets are not committed. Keep them locally:

```text
24-hour physiological monitoring/
Kaggle dataset/
```

## Clinical Boundary

This is a research and monitoring-support framework. It can help organise BP information and support ABPM prioritisation, but medication decisions must remain with the treating clinician.

> "Good monitoring should make complex physiology easier to understand, so care can reach people earlier and more clearly."
