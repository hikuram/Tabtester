from __future__ import annotations

import unittest

import pandas as pd

from tabtester.utils import align_feature_columns, regression_metrics, safe_stratify


class UtilsTest(unittest.TestCase):
    def test_align_feature_columns_preserves_expected_order(self):
        frame = pd.DataFrame({"b": [2], "a": [1], "extra": [3]})
        result = align_feature_columns(frame, ["a", "b"])
        self.assertEqual(list(result.columns), ["a", "b"])

    def test_align_feature_columns_rejects_missing_feature(self):
        frame = pd.DataFrame({"a": [1]})
        with self.assertRaises(ValueError):
            align_feature_columns(frame, ["a", "b"])

    def test_safe_stratify_requires_enough_rows(self):
        y = pd.Series(["a", "a", "b", "b"])
        self.assertIsNone(safe_stratify(y, 0.25))

    def test_regression_metrics_perfect_prediction(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(metrics["R2"], 1.0)
        self.assertAlmostEqual(metrics["RMSE"], 0.0)
        self.assertAlmostEqual(metrics["MAE"], 0.0)


if __name__ == "__main__":
    unittest.main()
