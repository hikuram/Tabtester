from __future__ import annotations

import io
import unittest

import pandas as pd

from tabtester.utils import (
    align_feature_columns,
    complete_target_columns,
    impute_with_backup,
    missing_column_summary,
    prepare_benchmark_target,
    read_csv,
    read_csv_with_info,
    regression_metrics,
    safe_stratify,
)


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

    def test_complete_target_columns_excludes_missing_columns(self):
        frame = pd.DataFrame({"complete": [1.0, 2.0], "missing": [1.0, None]})
        self.assertEqual(complete_target_columns(frame), ["complete"])

    def test_missing_column_summary_reports_counts(self):
        frame = pd.DataFrame({"a": [1.0, None], "b": [None, None], "c": [1.0, 2.0]})
        summary = missing_column_summary(frame)
        self.assertEqual(summary["Column"].tolist(), ["a", "b"])
        self.assertEqual(summary["Missing values"].tolist(), [1, 2])

    def test_prepare_benchmark_target_excludes_all_selected_targets(self):
        frame = pd.DataFrame(
            {
                "feature_a": [1, 2, 3],
                "feature_b": [4, 5, 6],
                "target_a": [10.0, 20.0, 30.0],
                "target_b": [100.0, 200.0, 300.0],
            }
        )

        X, y = prepare_benchmark_target(
            frame,
            "target_a",
            ["target_a", "target_b"],
            ["feature_b"],
            "Regression",
        )

        self.assertEqual(list(X.columns), ["feature_a"])
        self.assertEqual(y.tolist(), [10.0, 20.0, 30.0])

    def test_prepare_benchmark_target_rejects_non_numeric_regression_target(self):
        frame = pd.DataFrame(
            {
                "feature": [1, 2, 3],
                "target_good": [1.0, 2.0, 3.0],
                "target_bad": ["a", "b", "c"],
            }
        )

        with self.assertRaisesRegex(ValueError, "non-numeric"):
            prepare_benchmark_target(
                frame,
                "target_bad",
                ["target_good", "target_bad"],
                [],
                "Regression",
            )

    def test_prepare_benchmark_target_requires_two_classification_classes(self):
        frame = pd.DataFrame({"feature": [1, 2, 3], "target": ["a", "a", "a"]})

        with self.assertRaisesRegex(ValueError, "at least two classes"):
            prepare_benchmark_target(
                frame,
                "target",
                ["target"],
                [],
                "Classification",
            )

    def test_read_csv_rejects_cp932_when_japanese_support_is_disabled(self):
        text = "\u540d\u524d,\u5024\n\u30c6\u30b9\u30c8,1\n"
        uploaded = io.BytesIO(text.encode("cp932"))

        with self.assertRaises(ValueError):
            read_csv(uploaded)

    def test_read_csv_accepts_cp932_when_japanese_support_is_enabled(self):
        text = "\u540d\u524d,\u5024\n\u30c6\u30b9\u30c8,1\n"
        uploaded = io.BytesIO(text.encode("cp932"))

        frame = read_csv(uploaded, enable_japanese_support=True)

        self.assertEqual(frame.columns.tolist(), ["\u540d\u524d", "\u5024"])
        self.assertEqual(frame.iloc[0].tolist(), ["\u30c6\u30b9\u30c8", 1])

    def test_read_csv_auto_accepts_excel_style_utf16_tab_file(self):
        text = "name\tvalue\r\nalpha\t1\r\nbeta\t2\r\n"
        uploaded = io.BytesIO(text.encode("utf-16"))

        frame, encoding = read_csv_with_info(uploaded)

        self.assertEqual(encoding, "utf-16")
        self.assertEqual(frame.columns.tolist(), ["name", "value"])
        self.assertEqual(frame["value"].tolist(), [1, 2])

    def test_read_csv_manual_semicolon_delimiter(self):
        text = "name;value\nalpha;1\n"
        uploaded = io.BytesIO(text.encode("utf-8"))

        frame, encoding = read_csv_with_info(
            uploaded,
            encoding_mode="utf-8",
            delimiter_mode="semicolon",
        )

        self.assertEqual(encoding, "utf-8")
        self.assertEqual(frame.columns.tolist(), ["name", "value"])
        self.assertEqual(frame.iloc[0].tolist(), ["alpha", 1])

    def test_read_csv_manual_cp932_requires_japanese_support(self):
        text = "\u540d\u524d,\u5024\n\u30c6\u30b9\u30c8,1\n"
        uploaded = io.BytesIO(text.encode("cp932"))

        with self.assertRaisesRegex(ValueError, "ENABLE_JAPANESE_SUPPORT"):
            read_csv_with_info(uploaded, encoding_mode="cp932")

    def test_impute_with_backup_fills_target_and_preserves_original(self):
        frame = pd.DataFrame({"id": [1, 2, 3], "target": [10.0, None, 30.0], "x": [4, 5, 6]})
        missing_mask = frame["target"].isna()

        result, backup_column = impute_with_backup(frame, "target", missing_mask, [20.0])

        self.assertEqual(backup_column, "target__original")
        self.assertEqual(list(result.columns), ["id", "target", "target__original", "x"])
        self.assertEqual(result["target"].tolist(), [10.0, 20.0, 30.0])
        self.assertTrue(pd.isna(result.loc[1, "target__original"]))
        self.assertTrue(pd.isna(frame.loc[1, "target"]))

    def test_impute_with_backup_uses_numbered_name_on_collision(self):
        frame = pd.DataFrame(
            {
                "target": [1.0, None],
                "target__original": [100.0, 200.0],
            }
        )
        missing_mask = frame["target"].isna()

        result, backup_column = impute_with_backup(frame, "target", missing_mask, [2.0])

        self.assertEqual(backup_column, "target__original_2")
        self.assertEqual(result["target__original"].tolist(), [100.0, 200.0])
        self.assertEqual(result["target"].tolist(), [1.0, 2.0])

    def test_impute_with_backup_rejects_prediction_count_mismatch(self):
        frame = pd.DataFrame({"target": [None, None]})
        missing_mask = frame["target"].isna()

        with self.assertRaises(ValueError):
            impute_with_backup(frame, "target", missing_mask, [1.0])


if __name__ == "__main__":
    unittest.main()
