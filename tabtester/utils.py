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


def read_csv(uploaded_file: Any, enable_japanese_support: bool = False) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8"]
    if enable_japanese_support:
        encodings.append("cp932")
    for encoding in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    tried = ", ".join(encodings)
    raise ValueError(f"Could not decode CSV. Tried {tried}.")



def complete_target_columns(df: pd.DataFrame) -> list[str]:
    """Return columns with no missing values, preserving source order."""
    missing = df.isna().any(axis=0)
    return [str(column) for column in df.columns if not bool(missing[column])]


def prepare_benchmark_target(
    df: pd.DataFrame,
    target: str,
    benchmark_targets: list[str],
    excluded: list[str],
    task: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare one benchmark target while excluding every selected target."""
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    drop_columns: list[str] = []
    for column in [*benchmark_targets, *excluded]:
        if column in df.columns and column not in drop_columns:
            drop_columns.append(column)

    X = df.drop(columns=drop_columns).copy()
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after target and column exclusions.")

    y = df[target].copy()
    if task == "Regression":
        y_numeric = pd.to_numeric(y, errors="coerce")
        if y_numeric.isna().any():
            raise ValueError("Regression target contains non-numeric values.")
        y = y_numeric
    elif task == "Classification":
        if y.nunique(dropna=False) < 2:
            raise ValueError("Classification requires at least two classes.")
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return X, y


def missing_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize columns excluded from target selection due to missing values."""
    counts = df.isna().sum(axis=0)
    rows = []
    for column in df.columns:
        count = int(counts[column])
        if count == 0:
            continue
        rows.append(
            {
                "Column": str(column),
                "Missing values": count,
                "Missing %": 100.0 * count / max(len(df), 1),
            }
        )
    return pd.DataFrame(rows, columns=["Column", "Missing values", "Missing %"])


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def impute_with_backup(
    df: pd.DataFrame,
    target: str,
    missing_mask: pd.Series,
    predictions: Any,
) -> tuple[pd.DataFrame, str]:
    """Fill missing target values in place and preserve the original column."""
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    if len(missing_mask) != len(df):
        raise ValueError("Missing-value mask length does not match the dataset.")

    mask = pd.Series(missing_mask, index=df.index, dtype=bool)
    values = np.asarray(predictions)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("Imputation predictions must be one-dimensional.")
    if len(values) != int(mask.sum()):
        raise ValueError("Prediction count does not match the number of missing rows.")

    base_name = f"{target}__original"
    backup_column = base_name
    suffix = 2
    while backup_column in df.columns:
        backup_column = f"{base_name}_{suffix}"
        suffix += 1

    output = df.copy()
    target_position = output.columns.get_loc(target)
    output.insert(target_position + 1, backup_column, output[target].copy())
    output.loc[mask, target] = values
    return output, backup_column


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
