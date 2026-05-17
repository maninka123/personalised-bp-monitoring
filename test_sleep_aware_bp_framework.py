import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
