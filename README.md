# Sleep-Aware Blood Pressure Profiling Framework

What this project does:
This project converts raw 24-hour Ambulatory Blood Pressure Monitoring (ABPM) data into an interactive, sleep-aware patient profile for doctors.

Why it is useful for doctors:
ABPM devices capture detailed 24-hour data, which is difficult to interpret. This framework automates the feature extraction, flags risky patterns (like reverse dipping, morning surges, or high pulse pressure), and displays the patient's individual profile in an interactive web application.

How a new patient is analysed:
1. Patient wears a 24-hr device
2. Doctor uploads the standard CSV file (Time, Sys, Dia, HR)
3. Framework parses raw readings and identifies night/day (either by patient diary or actigraphy)
4. Framework generates the interactive clinical report.

Rule-Based Patient Pattern Flags:
The framework applies rules to flag high clinical risk patterns (e.g. Non-dipping, Reverse dipping, Morning surge, Isolated nocturnal hypertension).

This project builds a **sleep-aware blood pressure profiling framework** for personalised hypertension monitoring.

It uses two datasets, but they do different jobs:

- **Dryad dataset**: builds the actual sleep-aware BP framework from raw 24-hour ABPM readings.
- **Kaggle dataset**: trains and evaluates the machine-learning models using ABPM summary features.

Important: the ML models are trained **only on the Kaggle dataset**. Dryad and Kaggle are not merged row-by-row because they are different participant cohorts.

## Why Two Datasets?

The datasets complement each other:

| Dataset | Main role | Why it matters |
|---|---|---|
| Dryad 24-hour physiological monitoring | Framework development | Has raw ABPM readings, sleep/wake labels, HR, MAP, PP and participant metadata |
| Kaggle ABPM summary dataset | ML modelling | Has more rows and ready-made ABPM labels for model training |

Simple view:

```text
Dryad = builds the clinical framework
Kaggle = tests the machine-learning idea
```

More detailed view:

```text
Dryad raw 24-hour ABPM data
        |
        v
Clean invalid BP readings
        |
        v
Separate awake vs sleep BP
        |
        v
Calculate dipping, morning surge and BP variability
        |
        v
Create personalised BP profiles
        |
        v
Clinician-review monitoring recommendation
```

```text
Kaggle ABPM summary features
        |
        v
Use existing ABPM-derived features
        |
        v
Train logistic regression and random forest models
        |
        v
Predict abnormal ABPM labels
        |
        v
Save metrics, confusion matrices and model files
```

So the combined contribution is:

```text
Dryad explains the 24-hour physiology
        +
Kaggle supports the ML classification evidence
        =
Sleep-aware BP profiling framework with ML support
```

## Full Project Flow

```text
                +-----------------------------+
                |  Dryad sleep-aware ABPM     |
                |  raw BP + sleep/wake labels |
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |  BP feature extraction      |
                |  dipping, surge, variability|
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |  Personal BP profiles       |
                |  monitoring recommendation  |
                +-----------------------------+


                +-----------------------------+
                |  Kaggle ABPM summary data   |
                |  features + labels          |
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |  Machine-learning models    |
                |  Logistic Reg + RandomForest|
                +-------------+---------------+
                              |
                              v
                +-----------------------------+
                |  AUROC, F1, confusion matrix|
                |  saved .joblib models       |
                +-----------------------------+
```

## How a New Patient Is Handled

For a new patient, the framework plots the 24-hour BP curve, extracts sleep-aware BP features, compares the patient with clinically defined thresholds and reference distributions, and assigns an interpretable BP profile.

Machine learning is used only as a supporting analysis, not as the main decision method.

Example new-patient values:

| Feature | Value |
|---|---|
| Awake mean SBP | 140 mmHg |
| Sleep mean SBP | 138 mmHg |
| Dipping percentage | 1.4% |
| Morning surge | 24 mmHg |
| SBP variability | High |

Example output:

```text
Profile:
Non-dipper with morning surge and high variability

Review point:
Review night BP, sleep quality, adherence, caffeine or stress triggers,
and medication timing with clinician.
```

![New patient framework example](docs/figures/new_patient_framework_example.png)

**Figure. Interpretable new-patient BP profiling with ML support validation.**

The new-patient report is generated using rule-based sleep-aware ABPM features, including dipping percentage, morning surge and BP variability. The Kaggle ABPM dataset is used separately to test whether related ABPM feature groups can classify abnormal BP pattern labels. The ML component supports feature relevance but does not replace clinician judgement.

The figure shows four parts:

- line graph: how BP changes over 24 hours
- profile plot: where the patient lies compared with BP profile regions
- report card: what the clinician should review next
- separate ML support validation: whether similar ABPM feature groups can classify related BP pattern labels in Kaggle

ML support is not used to make the final clinical decision. It provides separate evidence that ABPM feature groups are useful for classifying related BP patterns.

Regenerate this figure with:

```bash
python scripts/create_new_patient_framework_figure.py
```

## Sleep-Aware BP Report Dashboard

The clinical prototype is a **doctor-first, patient-understandable dashboard**. It uses the rule-based Dryad-derived framework and hides machine-learning terms from the clinical interface.

The ML validation table is for the paper, README and research presentation only. It is not shown to doctors or patients in the dashboard.

```text
New patient ABPM file
        |
        v
Automatic sleep-aware feature calculation
        |
        v
Doctor dashboard
        |
        v
Patient-friendly report preview
        |
        v
PDF report for clinical review
```

Run the dashboard:

```bash
streamlit run sleep_aware_bp_report_app.py
```

The app includes an example patient, so it can be opened before uploading new data. For uploaded data, use a CSV or Excel file with at least:

```text
Time, Systolic, Diastolic
```

Optional columns:

```text
Day_Date, MAP, PP, HR, Wake_Sleep
```

If `Wake_Sleep` is missing, the app uses the sleep start and wake time entered in the sidebar.

Dashboard flow:

```text
Doctor dashboard
        |
        |-- summary cards: 24h BP, awake BP, sleep BP, dipping, surge, variability
        |-- 24-hour BP curve: systolic/diastolic BP, sleep shading, morning period
        |-- profile plot: sleep dipping % vs morning surge
        |-- pattern flags: non-dipper, morning surge, high variability, sustained high BP
        |-- review points: what the clinician should check next
```

Ask-about-report flow:

```text
Calculated BP report summary
        |
        v
Ask About This BP Report
        |
        |-- quick buttons: Explain profile, Why flagged, Explain to patient
        |-- custom question box
        |-- Gemma 4 explains the report using the saved Hugging Face token
        |
        v
Safe explanation only, not medication advice
```

The assistant receives only the calculated report summary, for example:

```json
{
  "profile": "Non-dipper with morning surge and high variability",
  "priority": "Review soon",
  "awake_bp": "140/86",
  "sleep_bp": "138/82",
  "dipping_percentage": "1.4%",
  "morning_surge": "24 mmHg",
  "bp_variability": "High",
  "review_points": [
    "Review night BP and sleep quality",
    "Review morning BP control",
    "Check stress, caffeine, adherence and measurement quality"
  ]
}
```

It does **not** send raw ABPM rows to the assistant.

No embedding database is needed here because the assistant answers from one small, structured report summary. This keeps the clinical explanation simple and reduces unnecessary data sharing.

Assistant model:

| Model | How it is used |
|---|---|
| Hugging Face Gemma 4 | The only visible assistant model in Streamlit, EXE/CLI and npm |

## Hugging Face Token Setup for Gemma

To use Gemma through Hugging Face, create a Hugging Face access token and use it as `HF_TOKEN`.

Recommended steps:

```text
1. Create or log in to a Hugging Face account.
2. Open Settings -> Access Tokens.
3. Create a token with read/inference access.
4. Accept the Gemma model terms on the Hugging Face model page if prompted.
5. Use the token in Streamlit, the EXE/CLI, or the npm app.
```

### Streamlit

Run the dashboard:

```bash
streamlit run sleep_aware_bp_report_app.py
```

Then open **Ask About This BP Report** and ask the question. There is no model selector or API key box in the app.

You can also set the token before starting Streamlit:

```powershell
$env:HF_TOKEN="hf_your_token_here"
streamlit run sleep_aware_bp_report_app.py
```

### Windows EXE / Python CLI

Save the Hugging Face token once:

```powershell
.\SleepAwareBPReportAssistant-v0.1.0-windows-x64.exe --save-token
```

Check token status:

```powershell
.\SleepAwareBPReportAssistant-v0.1.0-windows-x64.exe --token-status
```

Ask with Gemma:

```powershell
.\SleepAwareBPReportAssistant-v0.1.0-windows-x64.exe --question "Why is this patient flagged?"
```

For local Python:

```bash
python ask_bp_report.py --save-token
python ask_bp_report.py --question "Explain this to the patient"
```

### npm / Node companion app

The repo includes a small npm companion app in `npm_app/`.

Run the npm companion app:

```bash
cd npm_app
npm start
```

`npm start` opens an interactive menu where you can:

```text
1. Ask quick questions
2. Ask a custom question
3. View the report summary JSON sent to the assistant
4. Save or replace the Hugging Face token
```

Run one question directly:

```bash
npm run ask -- --question "Why is this patient flagged?"
```

Set token for one terminal session:

```powershell
$env:HF_TOKEN="hf_your_token_here"
npm run ask -- --question "Explain this to the patient"
```

Or save the token locally for future npm runs:

```bash
node src/ask-bp-report.mjs --save-token
node src/ask-bp-report.mjs --token-status
node src/ask-bp-report.mjs --question "What should the doctor review next?"
```

Safety rule:

```text
The assistant explains the BP report.
It does not diagnose, prescribe, or recommend medication changes.
```

Patient report flow:

```text
Patient report
        |
        |-- simple explanation of the BP pattern
        |-- simplified 24-hour curve
        |-- safe next steps
        |-- reminder not to change medication without the doctor
```

Exported PDF pages:

```text
Page 1: patient summary and overall profile
Page 2: BP graphs
Page 3: feature table
Page 4: doctor review checklist
```

## ML Support Validation

The new-patient profile is assigned using clear rule-based BP features. Separate Kaggle ML analysis shows that similar ABPM feature groups can classify related BP pattern labels.

| Rule-based feature | Kaggle ML target | What ML validates |
|---|---|---|
| Sleep BP fall / dipping % | `Circadian-Rythm` | Day-night BP features can identify abnormal rhythm |
| Morning surge | `Morning-Surge` | Wake-up BP features can identify morning rise |
| Overall high BP | `BP-Load` | 24h/day/night BP features can identify BP burden |
| Pulse pressure | `Pulse-Pressure` | Pressure-gap features can classify abnormal pulse pressure |
| BP variability | Feature group only | Variability helps model interpretation, but is not trained as a target here |

Do not interpret this as the ML model validating the new patient directly. The correct interpretation is:

```text
ML validates the relevance of ABPM feature groups
using a separate labelled dataset.
```

## BP Profiles

| Profile | Meaning |
|---|---|
| Normal dipper | Sleep SBP falls by 10-20% |
| Non-dipper | Sleep SBP fall is below 10% |
| Reverse dipper | Sleep SBP is higher than awake SBP |
| Extreme dipper | Sleep SBP falls by more than 20% |
| Morning surge | SBP rises after waking |
| Sustained high BP | BP remains high across day and night |

## What the Pipeline Produces

```text
Run script
   |
   v
outputs/
   |
   |-- Dryad cleaned BP readings
   |-- Dryad participant BP profiles
   |-- Kaggle ML metrics
   |-- Kaggle confusion matrices
   |-- Kaggle classification reports
   |-- Kaggle cross-validated predictions
   |-- saved ML models
   |-- generated figures
```

Main output files:

```text
outputs/
|-- dryad_participant_features.csv
|-- dryad_valid_bp_readings.csv
|-- kaggle_model_metrics.csv
|-- kaggle_confusion_matrices.csv
|-- kaggle_classification_reports.csv
|-- kaggle_cv_predictions.csv
|-- kaggle_feature_importance.csv
|-- analysis_summary.md
|-- models/
|   |-- *.joblib
|-- figures/
|   |-- confusion_matrices/
```

## Dataset Placement

Datasets are not committed to this repository. Place them beside `sleep_aware_bp_framework.py`:

```text
personalised-bp-monitoring/
|-- sleep_aware_bp_framework.py
|-- 24-hour physiological monitoring/
|   |-- Blood_Pressure_Sleep_Info.xlsx
|   |-- Participant_Information.csv
|   |-- Data_Collection_Notes.csv
|-- Kaggle dataset/
|   |-- y4dh3b3tfx-1/
|       |-- ABPM-dataset.arff
```

## Run

```bash
pip install -r requirements.txt
python sleep_aware_bp_framework.py
```

Run the clinical dashboard:

```bash
streamlit run sleep_aware_bp_report_app.py
```

Run the EXE-friendly command-line assistant:

```bash
python ask_bp_report.py
```

Ask one question directly:

```bash
python ask_bp_report.py --question "Why is this patient flagged?"
```

Run tests:

```bash
python -m unittest -v
```

## Machine-Learning Models

The ML section uses only the **Kaggle ABPM dataset**.

Models:

- logistic regression
- random forest

Targets:

- `Circadian-Rythm`
- `Pulse-Pressure`
- `BP-Load`
- `Morning-Surge`

For each target/model pair, the pipeline saves:

- model metrics
- cross-validated predictions
- confusion matrix values
- confusion matrix plots
- final `.joblib` model

## Clinical Boundary

This is a research and monitoring-support framework. It should not be used to automatically change antihypertensive medication.
