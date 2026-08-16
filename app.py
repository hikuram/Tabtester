from __future__ import annotations

import importlib.metadata
import io
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# TabICL imports
from tabicl import TabICLClassifier, TabICLRegressor

# TabFM imports
from tabfm import TabFMClassifier, TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0


APP_TITLE = "Tabtester"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 8
DEFAULT_BATCH_SIZE = 4
MIN_RECOMMENDED_ROWS = 300


@st.cache_resource(show_spinner="Loading TabFM base model weights...")
def load_tabfm_base_model(task: str, device: str) -> Any:
    """
    Loads and caches the TabFM foundation model weights to avoid reloading 
    into VRAM/RAM on every Streamlit interaction.
    """
    model_type = "regression" if task == "Regression" else "classification"
    target_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = tabfm_v1_0_0.load(model_type=model_type)
    model = model.to(target_device)
    return model


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
    model_name: str,
    task: str,
    device: str,
    random_state: int,
    # TabICLv2 specific kwargs
    n_estimators: int = 8,
    batch_size: int = 4,
    kv_cache: bool = False,
    model_path: str = "",
) -> Any:
    """Instantiates the selected model backend."""
    
    if model_name == "TabICLv2":
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
        
    elif model_name == "TabFM":
        base_model = load_tabfm_base_model(task, device)
        if task == "Regression":
            return TabFMRegressor(model=base_model)
        return TabFMClassifier(model=base_model)
    
    raise ValueError(f"Unknown model name: {model_name}")


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


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2 Score": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse
    }


def classification_metrics(model: Any, X_test: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred)
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            metrics["Log Loss"] = log_loss(y_true, proba, labels=model.classes_)
        except Exception:
            pass
    return metrics


def plot_regression(y_true: np.ndarray, preds_dict: dict[str, np.ndarray], target_name: str) -> plt.Figure:
    """Generates a scatter plot comparing actual vs predicted values for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 'X', 's']
    colors = sns.color_palette("husl", len(preds_dict))

    for (model_name, y_pred), marker, color in zip(preds_dict.items(), markers, colors):
        r2 = r2_score(y_true, y_pred)
        sns.scatterplot(
            x=y_true, y=y_pred, 
            label=f'{model_name} (R2={r2:.3f})', 
            marker=marker, color=color, s=70, alpha=0.7, ax=ax
        )

    # Plot perfect prediction diagonal line
    all_vals = np.concatenate([y_true] + list(preds_dict.values()))
    min_val, max_val = all_vals.min(), all_vals.max()
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs. Predicted Values ({target_name})")
    ax.legend()
    ax.grid(True)
    return fig


def plot_classification(y_true: np.ndarray, preds_dict: dict[str, np.ndarray], target_name: str) -> plt.Figure:
    """Generates side-by-side confusion matrices for all models."""
    n_models = len(preds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    cmaps = ['Blues', 'Oranges', 'Greens']

    for ax, (model_name, y_pred), cmap in zip(axes, preds_dict.items(), cmaps):
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
        ax.set_title(f"{model_name} (Accuracy: {acc:.3f})")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A lightweight workbench for testing and comparing tabular models (TabFM & TabICLv2).")

    with st.sidebar:
        st.subheader("Environment")
        st.text(f"tabfm: {package_version('tabfm')}")
        st.text(f"tabicl: {package_version('tabicl')}")
        st.text(f"streamlit: {package_version('streamlit')}")
        st.text(f"torch: {torch.__version__}")
        st.text(f"CUDA runtime: {torch.version.cuda or 'none'}")
        if torch.cuda.is_available():
            st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("CUDA GPU is not available to PyTorch.")
        
        st.divider()
        st.subheader("General Settings")
        
        execution_mode = st.radio(
            "Execution Mode", 
            ["TabFM Only", "TabICLv2 Only", "Compare Both"], 
            index=2,
            help="Select the backend model(s) to evaluate."
        )
        
        task = st.radio("Task", ["Regression", "Classification"], index=0)
        device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
        random_state = st.number_input("Random state", min_value=0, value=DEFAULT_RANDOM_STATE, step=1)
        
        # Determine active models based on execution mode
        if execution_mode == "TabFM Only":
            active_models = ["TabFM"]
        elif execution_mode == "TabICLv2 Only":
            active_models = ["TabICLv2"]
        else:
            active_models = ["TabFM", "TabICLv2"]

        # TabICL specific advanced settings
        n_estimators, batch_size, kv_cache, model_path = DEFAULT_N_ESTIMATORS, DEFAULT_BATCH_SIZE, False, ""
        if "TabICLv2" in active_models:
            with st.expander("TabICLv2 Advanced Settings", expanded=False):
                n_estimators = st.slider("Ensemble size", min_value=1, max_value=32, value=DEFAULT_N_ESTIMATORS)
                batch_size = st.slider("Batch size", min_value=1, max_value=32, value=DEFAULT_BATCH_SIZE)
                kv_cache = st.checkbox("Enable KV cache", value=False)
                model_path = st.text_input("Local checkpoint path", value="", placeholder="Blank = auto-download")

    # Upload Training Data
    train_file = st.file_uploader("Dataset CSV", type=["csv"], key="train_csv")
    if train_file is None:
        st.info("Upload a CSV dataset to begin.")
        return

    try:
        df = read_csv(train_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if df.empty or df.shape[1] < 2:
        st.error("The CSV must contain at least one row and two columns.")
        return
    if df.columns.duplicated().any():
        st.error("Duplicate column names are not supported. Rename duplicate columns first.")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    if len(df) < MIN_RECOMMENDED_ROWS:
        st.warning(
            "These models are generally pretrained on datasets with at least 300 training rows. "
            "Results on smaller datasets should be treated as experimental."
        )

    # Feature Selection
    target = st.selectbox("Target column", list(df.columns), index=len(df.columns) - 1)
    candidate_excluded = [c for c in df.columns if c != target]
    excluded = st.multiselect("Exclude feature columns (e.g., IDs, Names)", candidate_excluded, default=[])

    try:
        X, y, dropped_target_rows = prepare_xy(df, target, excluded)
    except Exception as exc:
        st.error(str(exc))
        return

    if len(X) < 5:
        st.error("At least 5 rows with a non-missing target are required.")
        return

    # Target Validation
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
        st.caption(f"Detected Classes: {n_classes}")

    st.caption(f"Features used: {X.shape[1]}")

    tab_eval, tab_predict, tab_impute = st.tabs(["Holdout Evaluation", "Predict New Rows", "Impute Missing Values"])

    # ---------------------------------------------------------
    # TAB 1: Holdout Evaluation
    # ---------------------------------------------------------
    with tab_eval:
        if dropped_target_rows:
            st.info(f"Note: {dropped_target_rows} rows with missing target values were ignored for this evaluation.")
            
        test_size = st.slider("Test fraction", min_value=0.05, max_value=0.5, value=DEFAULT_TEST_SIZE, step=0.05)
        
        if st.button("Run Holdout Evaluation", type="primary", use_container_width=True):
            stratify = safe_stratify(y, test_size) if task == "Classification" else None
            if task == "Classification" and stratify is None:
                st.warning("A stratified split was not possible; using a random split instead.")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
            )
            
            preds_dict = {}
            metrics_dict = {}
            
            for model_name in active_models:
                try:
                    with st.spinner(f"Running {model_name}..."):
                        model = build_model(
                            model_name=model_name, task=task, device=device, random_state=int(random_state),
                            n_estimators=n_estimators, batch_size=batch_size, 
                            kv_cache=kv_cache, model_path=model_path
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        preds_dict[model_name] = y_pred
                        
                        if task == "Regression":
                            metrics_dict[model_name] = regression_metrics(y_test, y_pred)
                        else:
                            metrics_dict[model_name] = classification_metrics(model, X_test, y_test, y_pred)
                
                except Exception as exc:
                    st.error(f"Error encountered while running {model_name}.")
                    st.exception(exc)
                    return

            st.subheader("Evaluation Metrics")
            metrics_df = pd.DataFrame(metrics_dict).round(4)
            st.dataframe(metrics_df, use_container_width=True)

            st.subheader("Visual Analysis")
            if task == "Regression":
                fig = plot_regression(np.asarray(y_test), preds_dict, target)
                st.pyplot(fig)
            else:
                fig = plot_classification(np.asarray(y_test), preds_dict, target)
                st.pyplot(fig)

            result_df = X_test.copy()
            result_df[target] = np.asarray(y_test)
            for m_name, y_pred in preds_dict.items():
                result_df[f"Prediction_{m_name}"] = y_pred
            
            st.download_button(
                "Download Evaluation Predictions",
                data=csv_bytes(result_df),
                file_name="tabtester_holdout_evaluation.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---------------------------------------------------------
    # TAB 2: Predict New Rows
    # ---------------------------------------------------------
    with tab_predict:
        pred_file = st.file_uploader("Prediction CSV", type=["csv"], key="pred_csv")
        if pred_file is None:
            st.info("Upload a CSV containing the same feature columns used for training to predict new rows.")
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
                    st.caption(f"Extra columns detected (will be ignored during prediction): {extra}")
                
                st.dataframe(pred_df.head(100), use_container_width=True)

                if st.button("Fit Full Data and Predict", type="primary", use_container_width=True):
                    out_df = pred_df.copy()
                    
                    for model_name in active_models:
                        try:
                            with st.spinner(f"Fitting {model_name} context and predicting..."):
                                model = build_model(
                                    model_name=model_name, task=task, device=device, random_state=int(random_state),
                                    n_estimators=n_estimators, batch_size=batch_size, 
                                    kv_cache=kv_cache, model_path=model_path
                                )
                                model.fit(X, y)
                                
                                X_pred = pred_df[required].copy()
                                pred = model.predict(X_pred)
                                out_df[f"Prediction_{model_name}"] = pred
                                
                                if task == "Classification" and hasattr(model, "predict_proba"):
                                    try:
                                        proba = model.predict_proba(X_pred)
                                        for idx, label in enumerate(model.classes_):
                                            out_df[f"Prob_{model_name}_{label}"] = proba[:, idx]
                                    except Exception:
                                        pass
                                        
                        except Exception as exc:
                            st.error(f"Error encountered while running {model_name}.")
                            st.exception(exc)
                            return

                    st.success("Prediction complete.")
                    st.dataframe(out_df.head(200), use_container_width=True)
                    st.download_button(
                        "Download Predictions",
                        data=csv_bytes(out_df),
                        file_name="tabtester_new_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    # ---------------------------------------------------------
    # TAB 3: Impute Missing Values
    # ---------------------------------------------------------
    with tab_impute:
        st.subheader("Impute Missing Values")
        
        missing_mask = df[target].isna()
        missing_count = int(missing_mask.sum())
        
        if missing_count == 0:
            st.info(f"The selected target column '{target}' has no missing values to impute.")
        else:
            st.write(f"**{missing_count}** missing values detected in '{target}'.")
            st.caption(
                "The model will use the rows without missing values as context (training data) "
                "to predict and fill the missing values in this column."
            )
            
            if st.button("Run Imputation", type="primary", use_container_width=True):
                out_df = df.copy()
                
                # Features for the missing rows (drop target and excluded columns)
                X_missing = df.loc[missing_mask].drop(columns=[target] + excluded)
                
                for model_name in active_models:
                    try:
                        with st.spinner(f"Fitting {model_name} context and imputing..."):
                            model = build_model(
                                model_name=model_name, task=task, device=device, random_state=int(random_state),
                                n_estimators=n_estimators, batch_size=batch_size, 
                                kv_cache=kv_cache, model_path=model_path
                            )
                            # Train on the clean data (X, y prepared earlier)
                            model.fit(X, y)
                            
                            # Predict for the missing rows
                            preds = model.predict(X_missing)
                            
                            # Create a new column with original data, replacing NaNs with predictions
                            imputed_col = f"{target}_imputed_by_{model_name}"
                            out_df[imputed_col] = out_df[target]
                            out_df.loc[missing_mask, imputed_col] = preds
                                    
                    except Exception as exc:
                        st.error(f"Error encountered while running {model_name}.")
                        st.exception(exc)
                        return

                st.success("Imputation complete.")
                
                st.write("Preview of Imputed Rows:")
                display_cols = [target] + [f"{target}_imputed_by_{m}" for m in active_models]
                st.dataframe(out_df.loc[missing_mask, display_cols].head(100), use_container_width=True)
                
                st.download_button(
                    "Download Imputed Dataset",
                    data=csv_bytes(out_df),
                    file_name="tabtester_imputed.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.divider()
    st.caption(
        "Note: TabFM relies on a foundational model with frozen weights, processing numeric and categorical data "
        "without requiring explicit encoding or imputation. TabICLv2 also supports in-context learning with similar properties."
    )


if __name__ == "__main__":
    main()
