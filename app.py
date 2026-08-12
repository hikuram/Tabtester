from __future__ import annotations

import importlib.metadata
import io
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier, TabICLRegressor


APP_TITLE = "Tabtester"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 8
DEFAULT_BATCH_SIZE = 4
MIN_RECOMMENDED_ROWS = 300


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def read_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not decode CSV. Tried utf-8-sig, utf-8, and cp932.")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def build_model(
    task: str,
    n_estimators: int,
    batch_size: int,
    kv_cache: bool,
    device: str,
    random_state: int,
    model_path: str,
) -> Any:
    kwargs = {
        "n_estimators": n_estimators,
        "batch_size": batch_size,
        "kv_cache": kv_cache,
        "device": None if device == "auto" else device,
        "random_state": random_state,
        "model_path": model_path.strip() or None,
        "use_amp": "auto",
        "offload_mode": "auto",
    }
    if task == "Regression":
        return TabICLRegressor(**kwargs)
    return TabICLClassifier(**kwargs)


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    excluded: list[str],
) -> tuple[pd.DataFrame, pd.Series, int]:
    before = len(df)
    clean = df.loc[df[target].notna()].copy()
    dropped = before - len(clean)
    feature_cols = [c for c in clean.columns if c != target and c not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns remain after exclusions.")
    X = clean[feature_cols].copy()
    y = clean[target].copy()
    return X, y, dropped


def safe_stratify(y: pd.Series, test_size: float) -> pd.Series | None:
    counts = y.value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    n_test = max(1, math.ceil(len(y) * test_size))
    n_train = len(y) - n_test
    if n_test < len(counts) or n_train < len(counts):
        return None
    return y


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return pd.DataFrame(
        {
            "Metric": ["R2", "MAE", "RMSE"],
            "Value": [r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), rmse],
        }
    )


def classification_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    names = ["Accuracy", "Balanced accuracy"]
    values = [accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)]
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            values.append(log_loss(y_true, proba, labels=model.classes_))
            names.append("Log loss")
        except Exception:
            pass
    return pd.DataFrame({"Metric": names, "Value": values})


def add_prediction_columns(model: Any, X_pred: pd.DataFrame, base: pd.DataFrame, task: str) -> pd.DataFrame:
    out = base.copy()
    pred = model.predict(X_pred)
    out["prediction"] = pred
    if task == "Classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_pred)
            for idx, label in enumerate(model.classes_):
                out[f"prob_{label}"] = proba[:, idx]
        except Exception:
            pass
    return out


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A lightweight workbench for testing tabular models. Current backend: TabICLv2.")

    with st.sidebar:
        st.subheader("Environment")
        st.text(f"tabicl: {package_version('tabicl')}")
        st.text(f"streamlit: {package_version('streamlit')}")
        st.text(f"torch: {torch.__version__}")
        st.text(f"CUDA runtime: {torch.version.cuda or 'none'}")
        if torch.cuda.is_available():
            st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("CUDA GPU is not available to PyTorch.")
        st.text(f"scikit-learn: {package_version('scikit-learn')}")
        st.text(f"pandas: {package_version('pandas')}")
        st.divider()
        st.subheader("Model settings")
        task = st.radio("Task", ["Regression", "Classification"], index=0)
        device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
        n_estimators = st.slider("Ensemble size", min_value=1, max_value=32, value=DEFAULT_N_ESTIMATORS)
        batch_size = st.slider("Batch size", min_value=1, max_value=32, value=DEFAULT_BATCH_SIZE)
        kv_cache = st.checkbox("Enable KV cache", value=False)
        random_state = st.number_input("Random state", min_value=0, value=DEFAULT_RANDOM_STATE, step=1)
        model_path = st.text_input("Local checkpoint path", value="", placeholder="Blank = auto-download")

    train_file = st.file_uploader("Training CSV", type=["csv"], key="train_csv")
    if train_file is None:
        st.info("Upload a training CSV to begin.")
        return

    try:
        df = read_csv(train_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if df.empty or df.shape[1] < 2:
        st.error("The training CSV must contain at least one row and two columns.")
        return
    if df.columns.duplicated().any():
        st.error("Duplicate column names are not supported. Rename duplicate columns first.")
        return

    st.subheader("Training data")
    st.dataframe(df.head(100), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    if len(df) < MIN_RECOMMENDED_ROWS:
        st.warning(
            "TabICLv2 was pretrained on datasets with at least 300 training rows, and the official project "
            "states that smaller datasets have not been tested. Results below 300 rows should be treated as experimental."
        )

    target = st.selectbox("Target column", list(df.columns), index=len(df.columns) - 1)
    candidate_excluded = [c for c in df.columns if c != target]
    excluded = st.multiselect("Exclude feature columns", candidate_excluded, default=[])

    try:
        X, y, dropped_target_rows = prepare_xy(df, target, excluded)
    except Exception as exc:
        st.error(str(exc))
        return

    if dropped_target_rows:
        st.warning(f"Dropped {dropped_target_rows} rows with missing target values.")
    if len(X) < 5:
        st.error("At least 5 rows with a non-missing target are required for this quick evaluation app.")
        return

    if task == "Regression":
        y_numeric = pd.to_numeric(y, errors="coerce")
        invalid = int(y_numeric.isna().sum())
        if invalid:
            st.error(f"Regression target contains {invalid} non-numeric values.")
            return
        y = y_numeric
    else:
        n_classes = y.nunique(dropna=False)
        if n_classes < 2:
            st.error("Classification requires at least two target classes.")
            return
        st.caption(f"Classes: {n_classes}")

    st.caption(f"Features used: {X.shape[1]}")

    tab_eval, tab_predict = st.tabs(["Holdout evaluation", "Predict new rows"])

    with tab_eval:
        test_size = st.slider("Test fraction", min_value=0.1, max_value=0.5, value=DEFAULT_TEST_SIZE, step=0.05)
        if st.button("Run holdout evaluation", type="primary", use_container_width=True):
            stratify = safe_stratify(y, test_size) if task == "Classification" else None
            if task == "Classification" and stratify is None:
                st.warning("A stratified split was not possible; using a random split instead.")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=int(random_state),
                stratify=stratify,
            )
            model = build_model(
                task=task,
                n_estimators=n_estimators,
                batch_size=batch_size,
                kv_cache=kv_cache,
                device=device,
                random_state=int(random_state),
                model_path=model_path,
            )
            try:
                with st.spinner("Running TabICLv2..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            except Exception as exc:
                st.exception(exc)
                return

            if task == "Regression":
                metrics = regression_metrics(y_test, y_pred)
                st.dataframe(metrics, hide_index=True, use_container_width=True)
                eval_df = pd.DataFrame({"actual": np.asarray(y_test), "predicted": y_pred})
                st.scatter_chart(eval_df, x="actual", y="predicted", use_container_width=True)
            else:
                metrics = classification_metrics(model, X_test, y_test, y_pred)
                st.dataframe(metrics, hide_index=True, use_container_width=True)
                labels = list(model.classes_) if hasattr(model, "classes_") else sorted(pd.unique(y))
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
                st.dataframe(cm_df, use_container_width=True)

            result_df = X_test.copy()
            result_df[target] = np.asarray(y_test)
            result_df["prediction"] = y_pred
            st.download_button(
                "Download evaluation predictions",
                data=csv_bytes(result_df),
                file_name="tabtester_evaluation.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_predict:
        pred_file = st.file_uploader("Prediction CSV", type=["csv"], key="pred_csv")
        if pred_file is None:
            st.info("Upload a CSV containing the same feature columns used for training.")
        else:
            try:
                pred_df = read_csv(pred_file)
            except Exception as exc:
                st.error(f"Failed to read prediction CSV: {exc}")
                return

            required = list(X.columns)
            missing = [c for c in required if c not in pred_df.columns]
            if missing:
                st.error(f"Missing required feature columns: {missing}")
            else:
                extra = [c for c in pred_df.columns if c not in required]
                if extra:
                    st.caption(f"Extra columns will be preserved but not used as features: {extra}")
                st.dataframe(pred_df.head(100), use_container_width=True)

                if st.button("Fit full data and predict", type="primary", use_container_width=True):
                    model = build_model(
                        task=task,
                        n_estimators=n_estimators,
                        batch_size=batch_size,
                        kv_cache=kv_cache,
                        device=device,
                        random_state=int(random_state),
                        model_path=model_path,
                    )
                    try:
                        with st.spinner("Fitting context and predicting..."):
                            model.fit(X, y)
                            out = add_prediction_columns(model, pred_df[required].copy(), pred_df, task)
                    except Exception as exc:
                        st.exception(exc)
                        return

                    st.success("Prediction complete.")
                    st.dataframe(out.head(200), use_container_width=True)
                    st.download_button(
                        "Download predictions",
                        data=csv_bytes(out),
                        file_name="tabtester_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    st.divider()
    st.caption(
        "The first TabICLv2 run downloads the public checkpoint unless a local checkpoint path is provided. "
        "For repeated internal use, pre-downloading the checkpoint can avoid external network access at runtime."
    )


if __name__ == "__main__":
    main()
