from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tabtester.backends.base import BackendConfig
from tabtester.backends.gaussian_process import GenericMaternGPBackend


class GenericMaternGPTest(unittest.TestCase):
    @staticmethod
    def _frame(n: int = 18) -> pd.DataFrame:
        x = np.linspace(0.0, 1.0, n)
        return pd.DataFrame(
            {
                "Feature A": x,
                "Feature B": 1.0 - x,
                "Category": np.where(np.arange(n) % 2 == 0, "A", "B"),
            }
        )

    def test_predicts_finite_values(self):
        X = self._frame()
        y = pd.Series(3.0 + 5.0 * np.linspace(0.0, 1.0, len(X)))
        config = BackendConfig(task="Regression", gp_max_iter=5)
        backend = GenericMaternGPBackend(config).fit(X.iloc[:-3], y.iloc[:-3])
        pred = backend.predict(X.iloc[-3:])
        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.isfinite(pred).all())
        self.assertIsNotNone(backend.final_loo_loss_)

    def test_handles_missing_and_categorical_features(self):
        X = self._frame()
        X.loc[2, "Feature A"] = np.nan
        X.loc[4, "Category"] = None
        y = pd.Series(2.0 + np.linspace(0.0, 1.0, len(X)))
        config = BackendConfig(task="Regression", gp_max_iter=3)
        backend = GenericMaternGPBackend(config).fit(X.iloc[:-2], y.iloc[:-2])
        pred = backend.predict(X.iloc[-2:])
        self.assertTrue(np.isfinite(pred).all())

    def test_is_regression_only(self):
        X = self._frame()
        y = pd.Series(np.arange(len(X)) % 2)
        config = BackendConfig(task="Classification", gp_max_iter=1)
        with self.assertRaisesRegex(ValueError, "Regression only"):
            GenericMaternGPBackend(config).fit(X, y)


if __name__ == "__main__":
    unittest.main()
