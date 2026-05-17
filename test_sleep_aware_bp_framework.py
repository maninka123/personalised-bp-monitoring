import unittest

import pandas as pd

from bp_report_assistant import answer_report_question, build_report_context
from clinical_report_utils import build_patient_profile, example_patient_abpm
from sleep_aware_bp_framework import classify_dipping, filter_valid_bp_readings


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
        from clinical_report_utils import prepare_patient_abpm

        valid = prepare_patient_abpm(raw)
        self.assertEqual(len(valid), 6)
        self.assertIn("Systolic", valid.columns)
        self.assertIn("Diastolic", valid.columns)
        self.assertIn("HR", valid.columns)


if __name__ == "__main__":
    unittest.main()
