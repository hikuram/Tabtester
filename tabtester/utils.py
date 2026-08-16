from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def read_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV. Tried utf-8-sig, utf-8, and cp932.")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def safe_stratify(y: pd.Series, test_size: float) -> pd.Series | None:
    counts = y.value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    n_test = max(1, math.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    if n_test < len(counts) or n_train < len(counts):
        return None
    return y


def align_feature_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df.loc[:, expected_columns].copy()


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "R2": float(r2_score(y_true_arr, y_pred_arr)),
        "RMSE": float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }


def classification_metrics(y_true, y_pred, probabilities=None, classes=None) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metrics = {
        "Accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
    }
    if probabilities is not None:
        try:
            metrics["Log Loss"] = float(log_loss(y_true_arr, probabilities, labels=classes))
        except ValueError:
            pass
    return metrics
