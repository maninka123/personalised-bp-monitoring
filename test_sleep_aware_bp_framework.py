import unittest

import pandas as pd

from bp_report_assistant import answer_report_question, build_report_context, token_status
from clinical_report_utils import (
    build_patient_profile,
    example_patient_abpm,
    extract_patient_details,
    prepare_patient_abpm,
)
from sleep_aware_bp_framework import classify_dipping, filter_valid_bp_readings, parse_measurement_datetime


class DippingClassificationTests(unittest.TestCase):
    def test_normal_dipper(self):
        dipping_pct, category = classify_dipping(120, 102, 6)
        self.assertAlmostEqual(dipping_pct, 15.0)
        self.assertEqual(category, "normal_dipper")

    def test_non_dipper(self):
        dipping_pct, category = classify_dipping(120, 114, 6)
        self.assertAlmostEqual(dipping_pct, 5.0)
        self.assertEqual(category, "non_dipper")

    def test_reverse_dipper(self):
        dipping_pct, category = classify_dipping(120, 126, 6)
        self.assertAlmostEqual(dipping_pct, -5.0)
        self.assertEqual(category, "reverse_dipper")

    def test_extreme_dipper(self):
        dipping_pct, category = classify_dipping(120, 90, 6)
        self.assertAlmostEqual(dipping_pct, 25.0)
        self.assertEqual(category, "extreme_dipper")

    def test_insufficient_sleep(self):
        dipping_pct, category = classify_dipping(120, 90, 1)
        self.assertTrue(pd.isna(dipping_pct))
        self.assertEqual(category, "insufficient_sleep")


class DataQualityTests(unittest.TestCase):
    def test_zero_bp_rows_are_removed(self):
        df = pd.DataFrame(
            {
                "Systolic": [120, 0, 135],
                "Diastolic": [80, 70, 0],
                "MAP": [93, 85, 90],
                "PP": [40, 30, 45],
                "HR": [70, 72, 75],
            }
        )
        valid = filter_valid_bp_readings(df)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid.iloc[0]["Systolic"], 120)

    def test_hh_mm_time_values_parse_with_dates(self):
        parsed = parse_measurement_datetime(
            pd.Series(["18/05/2026", "18/05/2026"]),
            pd.Series(["07:00", "08:30"]),
        )
        self.assertFalse(parsed.isna().any())
        self.assertEqual(parsed.iloc[0].hour, 7)
        self.assertEqual(parsed.iloc[1].minute, 30)

    def test_uploaded_file_falls_back_when_day_date_is_unparseable(self):
        raw = pd.DataFrame(
            {
                "Day_Date": ["not a date", "not a date", "not a date"],
                "Time": ["07:00", "08:00", "22:00"],
                "Systolic": [140, 142, 130],
                "Diastolic": [86, 88, 78],
                "HR": [72, 73, 65],
                "Wake_Sleep": [1, 1, 0],
            }
        )
        valid = prepare_patient_abpm(raw, patient_id="TEST")
        self.assertEqual(len(valid), 3)
        self.assertFalse(valid["measurement_datetime"].isna().any())


class ReportAssistantTests(unittest.TestCase):
    def test_report_context_uses_summary_not_raw_rows(self):
        profile = build_patient_profile(example_patient_abpm())
        context = build_report_context(profile)
        self.assertIn("profile", context)
        self.assertIn("review_points", context)
        self.assertNotIn("raw_readings", context)
        self.assertNotIn("measurement_datetime", context)

    def test_medication_change_question_is_guarded(self):
        profile = build_patient_profile(example_patient_abpm())
        context = build_report_context(profile)
        response = answer_report_question("Should I increase the medication dose?", context)
        self.assertIn("treating clinician", response.answer)
        self.assertEqual(response.source, "Safety guardrail")

    def test_uploaded_common_column_aliases_are_supported(self):
        raw = pd.DataFrame(
            {
                "Time": ["08:00", "09:00", "22:00", "23:00", "00:00", "01:00"],
                "Sys": [140, 138, 132, 130, 129, 131],
                "Dia": [85, 84, 78, 76, 75, 76],
                "Pulse": [72, 73, 68, 67, 66, 67],
            }
        )
        valid = prepare_patient_abpm(raw)
        self.assertEqual(len(valid), 6)
        self.assertIn("Systolic", valid.columns)
        self.assertIn("Diastolic", valid.columns)
        self.assertIn("HR", valid.columns)

    def test_patient_details_are_extracted_from_uploaded_file(self):
        raw = pd.DataFrame(
            {
                "Patient_ID": ["SP001", "SP001"],
                "Patient_Name": ["Sample Patient", "Sample Patient"],
                "Age": [55, 55],
                "Sex": ["Female", "Female"],
                "BMI": [28.2, 28.2],
                "ABPM_Date": ["2026-05-18", "2026-05-18"],
                "Time": ["07:00", "08:00"],
                "Systolic": [140, 142],
                "Diastolic": [86, 88],
            }
        )
        details = extract_patient_details(raw)
        self.assertEqual(details["Patient ID"], "SP001")
        self.assertEqual(details["Patient Name"], "Sample Patient")
        self.assertEqual(details["Age"], "55")
        self.assertEqual(details["Sex"], "Female")
        self.assertEqual(details["BMI"], "28.2")
        self.assertEqual(details["ABPM date"], "2026-05-18")

    def test_token_status_reports_gemma_provider(self):
        status = token_status()
        self.assertIn("Hugging Face Gemma 4", status)
        self.assertEqual(list(status.keys()), ["Hugging Face Gemma 4"])


if __name__ == "__main__":
    unittest.main()
