# Sleep-Aware BP Profiling and ABPM-TSL

![Status: Research](https://img.shields.io/badge/Status-Research-blue) ![Stage: Clinical_Prototype](https://img.shields.io/badge/Stage-Clinical_Prototype-brightgreen) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blueviolet)

**Tags:** hypertension monitoring, ABPM, nocturnal blood pressure, non-dipping, limited-input prediction, teacher-student learning, clinical decision support

This repository builds a research prototype for personalised hypertension monitoring. It has two connected parts:

1. A rule-based sleep-aware ABPM reporting framework that turns full 24-hour BP readings into clinician-readable profiles.
2. ABPM-TSL, a missingness-aware neural teacher-student model that estimates ABPM-defined risk patterns from limited clinic/home/demographic/lifestyle inputs when full ABPM is not immediately available.

The system supports clinical review and ABPM prioritisation. It does not diagnose hypertension without ABPM and does not recommend medication changes.

## Current Scope

```text
Full ABPM recording
  -> clean readings
  -> split awake and sleep periods
  -> compute 24h, awake and sleep BP features
  -> assign transparent rule-based profile
  -> clinician-facing report and PDF

Full ABPM sequence during model development
  -> self-supervised teacher model
  -> synthetic limited-input cohort
  -> missingness-aware student model
  -> Limited-Input Risk tab in desktop app
```

## Data Sources

| Data source | Role in the current repository | Important boundary |
|---|---|---|
| Primary 24-hour physiological monitoring dataset | Full ABPM profiling, rule-derived labels, teacher training, synthetic limited-input generation | Small cohort; not enough for final clinical claims |
| Synthetic limited-input cohort derived from primary ABPM data | ABPM-TSL student training, ablations, missingness tests | Method-development data, not clinical validation |
| Labelled ABPM summary dataset, previously called Kaggle/Mendeley support data | Classical baseline and feature-relevance support analysis | Uses ABPM-derived summary inputs, so it is not true no-ABPM deployment validation |

The old README described the machine-learning component mainly as logistic regression and random forest on the labelled ABPM summary dataset. That is no longer the main ML pathway. The current ML work is ABPM-TSL.

## Rule-Based ABPM Reporting

The rule-based pipeline reads ABPM rows, removes invalid zero values, separates awake and sleep readings, and calculates:

- 24-hour, awake and sleep SBP/DBP
- dipping percentage and dipping category
- morning surge
- BP variability
- pulse pressure and mean arterial pressure
- sleep BP data quality
- clinician-review flags

In the primary analysis:

| Result | Value |
|---|---:|
| Raw ABPM rows | 1,623 |
| Valid rows after zero filtering | 1,090 |
| Participants with valid ABPM | 30 |
| Normal dippers | 17 |
| Non-dippers | 7 |
| Extreme dippers | 3 |
| Insufficient sleep BP data | 3 |

The profile is assigned using transparent rules, not a neural model. Example rule-based outputs include normal dipper, non-dipper, reverse dipper, extreme dipper, high morning surge, high variability, raised sleep BP and sustained high BP burden.

## ABPM-TSL Neural Model

ABPM-TSL means **ABPM Teacher-Student Learning**.

During training, a teacher model sees the full ABPM sequence:

```text
SBP, DBP, HR, sleep/wake state, time features, observation masks
  -> CNN/GRU/Transformer teacher variants
  -> masked reconstruction pretraining
  -> multi-task ABPM phenotype prediction
```

The deployed student model sees only limited inputs:

```text
clinic BP
morning/evening home BP
3-day and 7-day home BP summaries
age, sex, BMI, resting HR
diabetes, smoking, previous hypertension, medication status
sleep duration, sleep quality, caffeine, alcohol, stress
missingness masks for all inputs
```

The student estimates five ABPM-defined risk probabilities:

- abnormal dipping
- high morning surge
- nocturnal hypertension
- high BP burden
- high variability

It also returns proxy regression targets such as dipping percentage, morning surge, sleep mean SBP, BP burden score and variability score.

## Current Neural Results

The reproducible neural pipeline is in:

```text
ABPM-TSL/scripts/run_abpm_tsl_pipeline.py
```

It generated:

- 2,400 synthetic limited-input rows
- 80 synthetic limited-input variants per participant
- teacher architecture ablations
- classical limited-input baselines
- student ablations
- missingness robustness tests
- TorchScript student model for the desktop app

Main held-out results from `ABPM-TSL/results/main_ablation_results.csv`:

| Model | Mean AUROC | Mean AUPRC | Mean F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Logistic regression | 0.686 | 0.466 | 0.422 | 0.676 |
| Random forest | 0.646 | 0.413 | 0.246 | 0.592 |
| HistGradientBoosting | 0.673 | 0.356 | 0.294 | 0.574 |
| Student MLP only | 0.722 | 0.536 | 0.343 | 0.643 |
| Student + feature dropout | 0.727 | 0.511 | 0.371 | 0.655 |
| Student + teacher soft labels | 0.730 | 0.511 | 0.377 | 0.656 |
| Full ABPM-TSL | 0.730 | 0.511 | 0.377 | 0.656 |

Per-target Full ABPM-TSL AUROC:

| Target | AUROC |
|---|---:|
| Abnormal dipping | 0.632 |
| Morning surge high | 0.517 |
| Nocturnal hypertension | 0.859 |
| High BP burden | 0.952 |
| High variability | 0.688 |

Missingness robustness:

| Missing rate | Full ABPM-TSL AUROC | Random forest AUROC |
|---:|---:|---:|
| 0.0 | 0.730 | 0.623 |
| 0.1 | 0.730 | 0.614 |
| 0.3 | 0.704 | 0.583 |
| 0.5 | 0.692 | 0.578 |
| 0.7 | 0.646 | 0.549 |

These results are method-development evidence on a synthetic limited-input cohort derived from ABPM data. They are not clinical validation.

## Desktop App

The Windows desktop app source is in:

```text
desktop_app/
```

It includes:

- Electron desktop UI
- FastAPI backend
- drag-and-drop ABPM upload
- patient report dashboard
- Gemma-assisted report explanation
- **Limited-Input Risk** tab
- TorchScript ABPM-TSL student model loading

The limited-input endpoint is:

```text
POST /api/limited-input-predict
```

When the trained model files are available, the endpoint uses:

```text
ABPM-TSL/models/student_abpm_tsl_torchscript.pt
ABPM-TSL/models/preprocessing.json
```

If the neural model cannot be loaded, the backend falls back to a transparent rule-like proof-of-concept estimator and reports the fallback reason in `model_status`.

Build the desktop app:

```bash
cd desktop_app
npm install
npm run dist-win
```

The installer is currently unsigned, so Windows may show an unknown-publisher warning.

## Streamlit Dashboard

Run the Streamlit clinical prototype:

```bash
pip install -r requirements.txt
streamlit run sleep_aware_bp_report_app.py
```

Uploaded ABPM files should include at least:

```text
Time, Systolic, Diastolic
```

Optional columns:

```text
Patient_ID, Patient_Name, Age, Sex, BMI, ABPM_Date,
Day_Date, MAP, PP, HR, Wake_Sleep
```

Sample fictional input files are in:

```text
Sample Patient Inputs/
desktop_app/sample-patient-inputs/
```

## Gemma Report Assistant

The **Ask About This BP Report** assistant explains only the calculated report summary. It does not receive raw ABPM rows and does not diagnose, prescribe or recommend medication changes.

Set a Hugging Face token as `HF_TOKEN`, or save it through the CLI:

```bash
python ask_bp_report.py --save-token
python ask_bp_report.py --question "Why is this patient flagged?"
```

## Repository Layout

```text
ABPM-TSL/
  data/                         synthetic limited-input dataset and ABPM sequences
  figures/                      neural method and result figures
  models/                       TorchScript student, state dicts and preprocessing metadata
  results/                      ablation, missingness and per-target result CSVs
  scripts/run_abpm_tsl_pipeline.py

desktop_app/
  src/                          Electron frontend
  python-backend/               FastAPI backend
  sample-patient-inputs/

paper/
  manuscript.docx
  manuscript_updated.docx
  manuscript.md
  figures/
  tables/

outputs/
  generated rule-based ABPM and old baseline outputs
```

Raw datasets are not committed. Place them locally like this:

```text
personalised-bp-monitoring/
|-- 24-hour physiological monitoring/
|   |-- Blood_Pressure_Sleep_Info.xlsx
|   |-- Participant_Information.csv
|   |-- Data_Collection_Notes.csv
|-- Kaggle dataset/
|   |-- y4dh3b3tfx-1/
|       |-- ABPM-dataset.arff
```

## Run Checks

```bash
python -m unittest -v
python sleep_aware_bp_framework.py
python ABPM-TSL/scripts/run_abpm_tsl_pipeline.py
```

The ABPM-TSL pipeline can take longer than the rule-based tests because it trains neural models and writes result artifacts.

## Clinical Boundary

This is a research and monitoring-support framework. It can help organise ABPM information, explain calculated report summaries, and prioritise ABPM review. It must not be used to automatically start, stop, increase, reduce or time antihypertensive medication. Medication decisions remain clinician-led and require full clinical context.
