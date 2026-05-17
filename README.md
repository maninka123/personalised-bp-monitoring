# Sleep-Aware Blood Pressure Profiling Framework 🩺

This repository contains a reproducible Python pipeline for a BP-first personalised hypertension monitoring study.

The framework uses 24-hour ambulatory blood pressure monitoring (ABPM) data with sleep/wake labels to identify interpretable circadian blood pressure profiles. A second ABPM summary dataset is used for lightweight machine-learning experiments.

The goal is decision support for clinician review, not automatic medication advice.

## What This Project Does

- Cleans sleep-annotated ABPM readings by removing failed zero-value measurements.
- Builds participant-level BP features from 24-hour readings.
- Detects clinically meaningful profiles such as normal dipping, non-dipping, extreme dipping, morning surge, high variability and sustained high BP.
- Uses participant metadata such as age group, sex, BMI, caffeine and alcohol intake.
- Runs baseline Kaggle ABPM classification models for circadian rhythm, pulse pressure, BP load and morning surge.
- Generates CSV outputs, model metrics, figures and a compact Markdown summary.

## Datasets

This repository intentionally does **not** include the datasets.

Place the datasets beside `sleep_aware_bp_framework.py` using this structure:

```text
personalised-bp-monitoring/
├── sleep_aware_bp_framework.py
├── 24-hour physiological monitoring/
│   ├── Blood_Pressure_Sleep_Info.xlsx
│   ├── Participant_Information.csv
│   ├── Data_Collection_Notes.csv
│   ├── Per_Participant_Sensor_Data/
│   └── Output_ECG_Segmentor_data/
└── Kaggle dataset/
    └── y4dh3b3tfx-1/
        └── ABPM-dataset.arff
```

Primary dataset:

- Dryad 24-hour physiological monitoring dataset.
- Used for raw ABPM curve analysis, sleep/wake separation and participant-level BP profiling.

Secondary dataset:

- Kaggle ABPM summary-feature dataset.
- Used for classification experiments only, not merged with Dryad participants.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run the Analysis 🚀

```bash
python sleep_aware_bp_framework.py
```

The pipeline writes results to `outputs/`.

Main outputs:

- `outputs/dryad_participant_features.csv`
- `outputs/dryad_valid_bp_readings.csv`
- `outputs/dryad_participant_features_sensitivity_no_device_issues.csv`
- `outputs/optional_physiology_coverage.csv`
- `outputs/kaggle_label_distribution.csv`
- `outputs/kaggle_model_metrics.csv`
- `outputs/kaggle_feature_importance.csv`
- `outputs/analysis_summary.md`
- `outputs/figures/`

## Methods in Brief

Dryad ABPM readings are filtered so rows with zero SBP, DBP, MAP or HR are excluded. Features are calculated per participant:

- 24-hour mean SBP/DBP
- awake and sleep mean SBP/DBP
- SBP dipping percentage
- morning surge
- SBP/DBP variability
- mean pulse pressure
- mean arterial pressure
- HR-SBP correlation

Dipping categories:

| Category | Rule |
|---|---|
| Normal dipper | 10-20% sleep SBP fall |
| Non-dipper | 0-<10% sleep SBP fall |
| Reverse dipper | Sleep SBP higher than awake SBP |
| Extreme dipper | >20% sleep SBP fall |
| Insufficient sleep | Fewer than 3 valid sleep BP readings |

Sustained high BP uses ABPM thresholds:

- 24-hour mean BP: `>=130/80`
- awake/daytime mean BP: `>=135/85`
- asleep/night-time mean BP: `>=120/70`

High variability and high morning surge are defined using the top quartile of the Dryad cohort, because the cohort is small.

## Kaggle Modelling

The Kaggle `.arff` file is used for baseline classification only. The pipeline trains:

- logistic regression
- random forest

Targets:

- `Circadian-Rythm`
- `Pulse-Pressure`
- `BP-Load`
- `Morning-Surge`

`BP-Variability` is excluded as a target because it is positive for every row in the dataset.

## Tests ✅

```bash
python -m unittest -v
```

The tests check:

- normal dipper classification
- non-dipper classification
- reverse dipper classification
- extreme dipper classification
- insufficient sleep handling
- zero BP row filtering

## Clinical Boundary

This framework is intended for research and clinician-review support. It should not be used to automatically increase, decrease or time antihypertensive medication.

## Project Status

This is an early research implementation. It is suitable for exploratory analysis, manuscript figures and reproducible feature extraction. Further external validation would be needed before clinical deployment.
