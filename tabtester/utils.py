from __future__ import annotations

import codecs
import io
import json
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


CSV_DELIMITERS: dict[str, str | None] = {
    "auto": None,
    "comma": ",",
    "tab": "\t",
    "semicolon": ";",
}

CSV_ENCODINGS = {
    "auto": None,
    "utf-8": "utf-8",
    "utf-8-sig": "utf-8-sig",
    "utf-16": "utf-16",
    "cp932": "cp932",
}


def _auto_csv_encodings(raw: bytes) -> list[str]:
    if raw.startswith(codecs.BOM_UTF8):
        return ["utf-8-sig"]
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return ["utf-32"]
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return ["utf-16"]

    return ["utf-8", "cp932"]


def read_csv_with_info(
    uploaded_file: Any,
    encoding_mode: str = "auto",
    delimiter_mode: str = "auto",
) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    if encoding_mode not in CSV_ENCODINGS:
        raise ValueError(f"Unsupported CSV encoding mode: {encoding_mode}")
    if delimiter_mode not in CSV_DELIMITERS:
        raise ValueError(f"Unsupported CSV delimiter mode: {delimiter_mode}")
    encodings = (
        _auto_csv_encodings(raw)
        if encoding_mode == "auto"
        else [CSV_ENCODINGS[encoding_mode]]
    )
    delimiter = CSV_DELIMITERS[delimiter_mode]
    errors: list[str] = []

    for encoding in encodings:
        if encoding is None:
            continue
        kwargs: dict[str, Any] = {"encoding": encoding}
        if delimiter is None:
            kwargs["sep"] = None
            kwargs["engine"] = "python"
        else:
            kwargs["sep"] = delimiter
        try:
            return pd.read_csv(io.BytesIO(raw), **kwargs), encoding
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")

    tried = ", ".join(encodings)
    detail = " | ".join(errors) if errors else "No encoding succeeded."
    raise ValueError(f"Could not read CSV. Tried {tried}. {detail}")


def read_csv(
    uploaded_file: Any,
    encoding_mode: str = "auto",
    delimiter_mode: str = "auto",
) -> pd.DataFrame:
    frame, _ = read_csv_with_info(
        uploaded_file,
        encoding_mode=encoding_mode,
        delimiter_mode=delimiter_mode,
    )
    return frame



def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _display_text(value: Any) -> Any:
    if _is_missing_scalar(value):
        return pd.NA
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        try:
            normalized = value.tolist() if isinstance(value, np.ndarray) else value
            if isinstance(normalized, set):
                normalized = sorted(normalized, key=str)
            return json.dumps(normalized, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only DataFrame with PyArrow-compatible column types.

    Streamlit serializes dataframes through PyArrow. Mixed pandas object
    columns can contain values that cannot be represented by one Arrow type.
    Preserve numeric, boolean, and datetime columns; convert ambiguous object
    or categorical columns to a consistent pandas string dtype for display.
    The source DataFrame is not modified.
    """
    safe = df.copy()
    for column in safe.columns:
        series = safe[column]
        dtype = series.dtype

        if isinstance(dtype, pd.CategoricalDtype):
            safe[column] = series.map(_display_text).astype("string")
            continue
        if pd.api.types.is_complex_dtype(dtype):
            safe[column] = series.map(_display_text).astype("string")
            continue
        if isinstance(dtype, (pd.PeriodDtype, pd.IntervalDtype)):
            safe[column] = series.map(_display_text).astype("string")
            continue
        if not pd.api.types.is_object_dtype(dtype):
            continue

        non_missing = [value for value in series.tolist() if not _is_missing_scalar(value)]
        if not non_missing:
            safe[column] = series.map(_display_text).astype("string")
            continue
        if all(isinstance(value, (bool, np.bool_)) for value in non_missing):
            safe[column] = series.astype("boolean")
            continue
        if all(
            isinstance(value, (int, float, complex, np.number))
            and not isinstance(value, (bool, np.bool_))
            for value in non_missing
        ):
            safe[column] = pd.to_numeric(series, errors="coerce")
            continue
        safe[column] = series.map(_display_text).astype("string")

    return safe


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
