from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "Patient_ID",
    "Patient_Name",
    "Age",
    "Sex",
    "BMI",
    "ABPM_Date",
    "Day_Date",
    "Time",
    "Systolic",
    "Diastolic",
    "MAP",
    "PP",
    "HR",
    "Wake_Sleep",
]

TIMES = [
    "00:00",
    "01:00",
    "02:00",
    "03:00",
    "04:00",
    "05:00",
    "06:00",
    "07:00",
    "08:00",
    "09:00",
    "10:00",
    "12:00",
    "14:00",
    "16:00",
    "18:00",
    "20:00",
    "22:00",
    "23:00",
]

WAKE_SLEEP = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]


PATIENTS = [
    {
        "file": "sample_01_normal_dipper.csv",
        "id": "SP001",
        "name": "Sample Patient Normal Dipper",
        "age": 46,
        "sex": "Female",
        "bmi": 24.8,
        "sbp": [110, 108, 106, 105, 106, 108, 110, 120, 121, 125, 125, 124, 126, 128, 127, 124, 112, 110],
        "dbp": [68, 67, 66, 65, 66, 67, 68, 74, 75, 78, 78, 77, 78, 79, 78, 76, 70, 69],
        "hr": [58, 57, 56, 56, 57, 58, 60, 69, 70, 73, 72, 70, 71, 72, 70, 68, 61, 59],
    },
    {
        "file": "sample_02_non_dipper_morning_surge.csv",
        "id": "SP002",
        "name": "Sample Patient Non Dipper",
        "age": 59,
        "sex": "Male",
        "bmi": 29.1,
        "sbp": [136, 138, 137, 135, 136, 139, 140, 160, 162, 154, 148, 145, 144, 143, 145, 146, 139, 138],
        "dbp": [82, 83, 82, 81, 82, 83, 84, 96, 97, 91, 88, 86, 85, 85, 86, 87, 83, 82],
        "hr": [63, 62, 61, 61, 62, 64, 66, 84, 86, 80, 76, 74, 73, 72, 74, 75, 66, 64],
    },
    {
        "file": "sample_03_reverse_dipper.csv",
        "id": "SP003",
        "name": "Sample Patient Reverse Dipper",
        "age": 64,
        "sex": "Female",
        "bmi": 31.2,
        "sbp": [148, 150, 151, 153, 152, 154, 153, 142, 144, 143, 141, 140, 139, 141, 142, 143, 149, 150],
        "dbp": [88, 89, 90, 91, 90, 92, 91, 84, 85, 84, 83, 82, 82, 83, 84, 85, 89, 90],
        "hr": [66, 65, 64, 64, 65, 67, 69, 72, 74, 73, 72, 71, 70, 71, 72, 73, 67, 66],
    },
    {
        "file": "sample_04_extreme_dipper.csv",
        "id": "SP004",
        "name": "Sample Patient Extreme Dipper",
        "age": 52,
        "sex": "Male",
        "bmi": 26.5,
        "sbp": [94, 92, 91, 90, 91, 93, 95, 128, 132, 130, 129, 127, 128, 130, 131, 128, 98, 96],
        "dbp": [58, 57, 56, 55, 56, 57, 58, 80, 82, 81, 80, 79, 80, 81, 82, 79, 60, 59],
        "hr": [54, 53, 52, 52, 53, 54, 56, 74, 76, 75, 74, 73, 74, 75, 76, 73, 57, 55],
    },
    {
        "file": "sample_05_sustained_high_bp.csv",
        "id": "SP005",
        "name": "Sample Patient Sustained High BP",
        "age": 67,
        "sex": "Female",
        "bmi": 33.4,
        "sbp": [144, 145, 146, 145, 144, 147, 148, 152, 155, 154, 153, 151, 150, 152, 154, 153, 146, 145],
        "dbp": [86, 87, 88, 87, 86, 88, 89, 91, 93, 92, 91, 90, 89, 90, 92, 91, 87, 86],
        "hr": [64, 63, 62, 62, 63, 65, 67, 77, 79, 78, 77, 75, 74, 75, 77, 76, 66, 64],
    },
    {
        "file": "sample_06_high_variability.csv",
        "id": "SP006",
        "name": "Sample Patient High Variability",
        "age": 55,
        "sex": "Male",
        "bmi": 28.7,
        "sbp": [118, 142, 112, 150, 121, 146, 126, 166, 138, 158, 132, 170, 125, 155, 135, 162, 140, 119],
        "dbp": [72, 86, 68, 88, 74, 87, 77, 96, 84, 92, 80, 98, 76, 90, 82, 94, 86, 73],
        "hr": [58, 66, 56, 68, 60, 67, 62, 84, 72, 80, 70, 86, 66, 78, 72, 82, 68, 59],
    },
    {
        "file": "sample_07_morning_surge.csv",
        "id": "SP007",
        "name": "Sample Patient Morning Surge",
        "age": 49,
        "sex": "Female",
        "bmi": 25.9,
        "sbp": [114, 112, 111, 110, 112, 113, 114, 146, 150, 140, 132, 128, 127, 129, 130, 128, 116, 115],
        "dbp": [69, 68, 67, 66, 67, 68, 69, 89, 92, 86, 81, 78, 77, 78, 79, 78, 70, 69],
        "hr": [57, 56, 55, 55, 56, 57, 59, 82, 85, 78, 73, 70, 69, 70, 71, 69, 60, 58],
    },
    {
        "file": "sample_08_sleep_hypertension.csv",
        "id": "SP008",
        "name": "Sample Patient Sleep Hypertension",
        "age": 62,
        "sex": "Male",
        "bmi": 30.6,
        "sbp": [128, 130, 132, 131, 130, 133, 134, 138, 140, 139, 137, 136, 135, 136, 137, 138, 131, 129],
        "dbp": [76, 77, 78, 77, 77, 79, 80, 84, 85, 84, 83, 82, 81, 82, 83, 84, 78, 77],
        "hr": [60, 59, 58, 58, 59, 61, 63, 74, 76, 75, 74, 72, 71, 72, 73, 74, 62, 60],
    },
    {
        "file": "sample_09_borderline_control.csv",
        "id": "SP009",
        "name": "Sample Patient Borderline",
        "age": 41,
        "sex": "Female",
        "bmi": 23.7,
        "sbp": [116, 115, 114, 113, 114, 116, 118, 128, 129, 134, 132, 131, 130, 132, 133, 131, 119, 117],
        "dbp": [70, 69, 68, 68, 68, 69, 71, 80, 81, 84, 82, 81, 80, 81, 82, 81, 72, 71],
        "hr": [56, 55, 54, 54, 55, 56, 58, 69, 70, 73, 71, 70, 69, 70, 71, 69, 59, 57],
    },
    {
        "file": "sample_10_limited_sleep_data.csv",
        "id": "SP010",
        "name": "Sample Patient Limited Sleep Data",
        "age": 70,
        "sex": "Male",
        "bmi": 27.8,
        "times": ["07:00", "08:00", "09:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "23:00"],
        "wake_sleep": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "sbp": [152, 154, 151, 149, 148, 150, 151, 153, 150, 142],
        "dbp": [90, 92, 89, 88, 87, 88, 89, 91, 88, 84],
        "hr": [76, 78, 75, 74, 73, 74, 75, 77, 74, 66],
    },
]


def row_for(patient: dict, idx: int, time: str, wake_sleep: int) -> dict[str, object]:
    sbp = patient["sbp"][idx]
    dbp = patient["dbp"][idx]
    return {
        "Patient_ID": patient["id"],
        "Patient_Name": patient["name"],
        "Age": patient["age"],
        "Sex": patient["sex"],
        "BMI": patient["bmi"],
        "ABPM_Date": "2026-05-18",
        "Day_Date": "18/05/2026",
        "Time": time,
        "Systolic": sbp,
        "Diastolic": dbp,
        "MAP": round(dbp + (sbp - dbp) / 3, 1),
        "PP": sbp - dbp,
        "HR": patient["hr"][idx],
        "Wake_Sleep": wake_sleep,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_template(output_dir: Path) -> None:
    rows = [
        {
            "Patient_ID": "PAT001",
            "Patient_Name": "Example Patient",
            "Age": "55",
            "Sex": "Female",
            "BMI": "28.0",
            "ABPM_Date": "2026-05-18",
            "Day_Date": "18/05/2026",
            "Time": "22:00",
            "Systolic": "132",
            "Diastolic": "78",
            "MAP": "96.0",
            "PP": "54",
            "HR": "66",
            "Wake_Sleep": "0",
        },
        {
            "Patient_ID": "PAT001",
            "Patient_Name": "Example Patient",
            "Age": "55",
            "Sex": "Female",
            "BMI": "28.0",
            "ABPM_Date": "2026-05-18",
            "Day_Date": "18/05/2026",
            "Time": "23:00",
            "Systolic": "130",
            "Diastolic": "77",
            "MAP": "94.7",
            "PP": "53",
            "HR": "64",
            "Wake_Sleep": "0",
        },
        {
            "Patient_ID": "PAT001",
            "Patient_Name": "Example Patient",
            "Age": "55",
            "Sex": "Female",
            "BMI": "28.0",
            "ABPM_Date": "2026-05-18",
            "Day_Date": "18/05/2026",
            "Time": "07:00",
            "Systolic": "146",
            "Diastolic": "88",
            "MAP": "107.3",
            "PP": "58",
            "HR": "78",
            "Wake_Sleep": "1",
        },
        {
            "Patient_ID": "PAT001",
            "Patient_Name": "Example Patient",
            "Age": "55",
            "Sex": "Female",
            "BMI": "28.0",
            "ABPM_Date": "2026-05-18",
            "Day_Date": "18/05/2026",
            "Time": "08:00",
            "Systolic": "150",
            "Diastolic": "90",
            "MAP": "110.0",
            "PP": "60",
            "HR": "82",
            "Wake_Sleep": "1",
        },
    ]
    write_csv(output_dir / "TEMPLATE_ABPM_with_patient_details.csv", rows)


def write_readme(output_dir: Path) -> None:
    text = """# Sample Patient Inputs

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
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def create_samples(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_template(output_dir)
    write_readme(output_dir)
    for patient in PATIENTS:
        times = patient.get("times", TIMES)
        wake_sleep = patient.get("wake_sleep", WAKE_SLEEP)
        rows = [row_for(patient, idx, time, wake_sleep[idx]) for idx, time in enumerate(times)]
        write_csv(output_dir / patient["file"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Sample Patient Inputs",
        help="Folder where the sample/template ABPM files will be written.",
    )
    args = parser.parse_args()
    create_samples(args.output)
    print(f"Wrote sample patient input files to {args.output}")


if __name__ == "__main__":
    main()
