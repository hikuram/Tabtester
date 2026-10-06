from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabtester.backends.base import BackendConfig
from tabtester.backends.foundation import KumoTabularBackend


class _FakeTable:
    def __init__(self, df: pd.DataFrame, stypes: dict[str, object], device: str) -> None:
        self.df = df.copy()
        self.stypes = stypes
        self.device = device

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, stypes: dict[str, object], device: str):
        return cls(df, stypes, device)


class _FakeOutput:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def cpu(self):
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


class _FakeKumo:
    predict_calls = 0
    last_init: tuple[str, str, str] | None = None
    last_fit: dict[str, object] | None = None

    def __init__(self, *, task: str, size: str, device: str) -> None:
        type(self).last_init = (task, size, device)

    def fit(self, **kwargs) -> None:
        type(self).last_fit = kwargs

    def predict(self, *, x: _FakeTable):
        type(self).predict_calls += 1
        return _FakeOutput(
            pd.DataFrame(
                [[0.2, 0.8], [0.7, 0.3]],
                columns=["A", "B"],
                index=x.df.index,
            )
        )


class KumoBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeKumo.predict_calls = 0
        _FakeKumo.last_init = None
        _FakeKumo.last_fit = None

    def _fake_sdm(self) -> types.ModuleType:
        module = types.ModuleType("sdm")
        module.TableTensor = _FakeTable
        module.models = types.SimpleNamespace(KumoTabular=_FakeKumo)

        def infer_stypes(frame: pd.DataFrame, **kwargs):
            self.assertEqual(kwargs.get("_low_cardinality"), "infer")
            return {
                column: ("numerical" if pd.api.types.is_numeric_dtype(frame[column]) else "categorical")
                for column in frame.columns
            }

        module.infer_stypes = infer_stypes
        return module

    def test_classification_fit_predict_and_probability_cache(self):
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "group": ["a", "a", "b", "b"]})
        y = pd.Series(["A", "B", "A", "B"], name="target")
        config = BackendConfig(
            task="Classification",
            device="cpu",
            random_state=7,
            kumo_size="small",
            kumo_n_estimators=3,
        )
        backend = KumoTabularBackend(config)

        query = X.iloc[:2].copy()
        with patch.dict(sys.modules, {"sdm": self._fake_sdm()}):
            backend.fit(X, y)
            pred = backend.predict(query)
            proba = backend.predict_proba(query)

        self.assertEqual(_FakeKumo.last_init, ("classification", "small", "cpu"))
        self.assertEqual(_FakeKumo.last_fit["num_estimators"], 3)
        self.assertEqual(_FakeKumo.last_fit["estimator_batch_size"], 1)
        np.testing.assert_array_equal(pred, np.array(["B", "A"]))
        np.testing.assert_allclose(proba, np.array([[0.2, 0.8], [0.7, 0.3]]))
        np.testing.assert_array_equal(backend.class_labels(), np.array(["A", "B"]))
        self.assertEqual(_FakeKumo.predict_calls, 1)

    def test_rejects_more_than_ten_classes(self):
        X = pd.DataFrame({"x": np.arange(11)})
        y = pd.Series([f"c{i}" for i in range(11)])
        backend = KumoTabularBackend(BackendConfig(task="Classification", device="cpu"))
        with patch.dict(sys.modules, {"sdm": self._fake_sdm()}):
            with self.assertRaisesRegex(ValueError, "at most 10 classes"):
                backend.fit(X, y)


if __name__ == "__main__":
    unittest.main()
