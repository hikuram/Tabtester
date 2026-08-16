from __future__ import annotations

import importlib.metadata
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.model_selection import train_test_split

from tabtester.backends import (
    BackendConfig,
    available_model_names,
    foundation_model_names,
    make_backend,
)
from tabtester.plotting import plot_classification, plot_regression
from tabtester.utils import (
    align_feature_columns,
    classification_metrics,
    csv_bytes,
    read_csv,
    regression_metrics,
    safe_stratify,
)

APP_TITLE = "Tabtester"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
TABFM_LICENSE_NOTICE = (
    "TabFM default pretrained weights are licensed separately from the TabFM source code. "
    "They are restricted to non-commercial, non-production use."
)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def build_config(
    task: str,
    device: str,
    random_state: int,
    n_trials: int,
    time_budget: int,
    tabicl_n_estimators: int,
    tabicl_batch_size: int,
    tabicl_kv_cache: bool | str,
    tabicl_use_amp: bool | str,
    tabicl_offload_mode: bool | str,
    tabfm_checkpoint_path: str | None,
) -> BackendConfig:
    return BackendConfig(
        task=task,
        device=device,
        random_state=random_state,
        n_trials=n_trials,
        time_budget=time_budget,
        tabicl_n_estimators=tabicl_n_estimators,
        tabicl_batch_size=tabicl_batch_size,
        tabicl_kv_cache=tabicl_kv_cache,
        tabicl_use_amp=tabicl_use_amp,
        tabicl_offload_mode=tabicl_offload_mode,
        tabfm_checkpoint_path=tabfm_checkpoint_path or None,
    )


def generate_shap_plot(model: Any, X_data: pd.DataFrame, title: str):
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data, show=False)
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    except Exception as exc:
        st.warning(f"Could not generate SHAP plot: {exc}")
        return None


def render_environment_status() -> None:
    st.subheader("Environment")
    st.text(f"Python packages: torch {package_version('torch')}")
    st.text(f"TabICL: {package_version('tabicl')}")
    st.text(f"TabFM: {package_version('tabfm')}")
    st.text(f"CUDA runtime: {torch.version.cuda or 'None'}")
    st.text(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.text(f"GPU: {torch.cuda.get_device_name(0)}")
        st.text(f"BF16: {torch.cuda.is_bf16_supported()}")


def validate_tabfm_use(
    model_names: list[str],
    acknowledged: bool,
    checkpoint_path: str | None,
) -> bool:
    if "TabFM" not in model_names or checkpoint_path:
        return True
    if acknowledged:
        return True
    st.error("Acknowledge the TabFM pretrained-weight license notice before downloading or using the default TabFM weights.")
    return False


def render_primary_metric_plot(results: pd.DataFrame, task: str) -> None:
    metric = "R2" if task == "Regression" else "Accuracy"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results["Model"], results[metric])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_time_plot(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results))
    fit_times = results["Fit Time (s)"].to_numpy()
    predict_times = results["Predict Time (s)"].to_numpy()
    ax.bar(x, fit_times, label="Fit/prep")
    ax.bar(x, predict_times, bottom=fit_times, label="Predict")
    ax.set_xticks(x, results["Model"], rotation=45, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Execution time")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("A lightweight workbench for comparing tabular foundation models and classical baselines.")

    available_models = available_model_names()
    available_foundation_models = foundation_model_names()

    with st.sidebar:
        render_environment_status()
        st.divider()
        st.subheader("Benchmark settings")
        task = st.radio("Task type", ["Regression", "Classification"], index=0)
        default_models = [name for name in ["TabICLv2", "XGBoost (Default)"] if name in available_models]
        if not default_models and available_models:
            default_models = available_models[:1]
        selected_models = st.multiselect(
            "Models to benchmark",
            available_models,
            default=default_models,
        )

        device_options = ["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        device = st.selectbox("Foundation model device", device_options, index=0)
        random_state = int(st.number_input("Random seed", min_value=0, value=DEFAULT_RANDOM_STATE, step=1))

        if "TabFM" in available_models:
            st.divider()
            st.subheader("TabFM license")
            st.warning(TABFM_LICENSE_NOTICE)
            tabfm_ack = st.checkbox("I acknowledge the TabFM pretrained-weight license restriction.")
        else:
            tabfm_ack = False

        with st.expander("Foundation model settings", expanded=False):
            tabicl_n_estimators = st.slider("TabICLv2 estimators", 1, 16, 8, 1)
            tabicl_batch_size = st.select_slider("TabICLv2 batch size", options=[1, 2, 4, 8, 16], value=4)
            kv_cache_label = st.selectbox("TabICLv2 KV cache", ["off", "repr", "kv"], index=0)
            tabicl_kv_cache: bool | str = False if kv_cache_label == "off" else kv_cache_label
            tabicl_use_amp = st.selectbox("TabICLv2 AMP", ["auto", True, False], index=0)
            tabicl_offload_mode = st.selectbox("TabICLv2 offload", ["auto", True, False], index=0)
            tabfm_checkpoint_path = st.text_input(
                "TabFM local checkpoint path",
                value=os.getenv("TABFM_CHECKPOINT_PATH", ""),
                help="Leave empty to use the default Hugging Face checkpoint.",
            ).strip()

        with st.expander("AutoML and tuning", expanded=False):
            n_trials = st.slider("Optuna trials", min_value=5, max_value=50, value=10, step=5)
            time_budget = st.slider("AutoML time budget (seconds)", min_value=10, max_value=300, value=30, step=10)

    if not available_models:
        st.error("No model backend is available. Install at least one supported backend.")
        return

    train_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    if train_file is None:
        st.info("Upload a CSV dataset to begin.")
        return

    try:
        df = read_csv(train_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if df.empty or len(df.columns) < 2:
        st.error("The dataset must contain at least one feature column and one target column.")
        return
    if df.columns.duplicated().any():
        st.error("Duplicate column names are not supported.")
        return

    st.subheader("Data preview")
    st.dataframe(df.head(100), use_container_width=True)

    target = st.selectbox("Target column", list(df.columns), index=len(df.columns) - 1)
    candidate_excluded = [column for column in df.columns if column != target]
    excluded = st.multiselect("Exclude columns", candidate_excluded, default=[])

    df_clean = df.dropna(subset=[target]).copy()
    dropped_rows = len(df) - len(df_clean)
    if dropped_rows:
        st.caption(f"Rows with missing target excluded from training/evaluation: {dropped_rows}")

    X = df_clean.drop(columns=[target] + excluded)
    y = df_clean[target].copy()
    if X.shape[1] == 0:
        st.error("No feature columns remain after exclusions.")
        return

    if task == "Regression":
        y_numeric = pd.to_numeric(y, errors="coerce")
        if y_numeric.isna().any():
            st.error("Regression target contains non-numeric values.")
            return
        y = y_numeric
    else:
        st.caption(f"Classes: {y.nunique(dropna=False)}")
        if y.nunique(dropna=False) < 2:
            st.error("Classification requires at least two classes.")
            return

    config = build_config(
        task,
        device,
        random_state,
        n_trials,
        time_budget,
        tabicl_n_estimators,
        tabicl_batch_size,
        tabicl_kv_cache,
        tabicl_use_amp,
        tabicl_offload_mode,
        tabfm_checkpoint_path,
    )

    tab_eval, tab_predict, tab_impute = st.tabs(
        ["Benchmark and Evaluation", "Predict New Rows", "Impute Missing Target"]
    )

    with tab_eval:
        st.markdown("### Model performance benchmark")
        test_size = st.slider("Test fraction", 0.05, 0.5, DEFAULT_TEST_SIZE, 0.05)

        if "TabICLv2" in selected_models and len(X) < 300:
            st.warning("TabICLv2 was pretrained on datasets starting around 300 rows; smaller datasets should be treated as an empirical test case.")
        if "TabFM" in selected_models and len(X) > 100:
            st.info("TabFM uses a bounded context window and samples context rows when the training table is larger than its configured context size.")
        if task == "Classification" and "TabFM" in selected_models and y.nunique() > 10:
            st.warning("TabFM v1.0.0 supports at most 10 classes and will not run for this target.")

        if st.button("Run benchmark", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("Select at least one model.")
            elif validate_tabfm_use(selected_models, tabfm_ack, tabfm_checkpoint_path):
                stratify = safe_stratify(y, test_size) if task == "Classification" else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )

                results: list[dict[str, Any]] = []
                predictions: dict[str, np.ndarray] = {}
                shap_payload = None
                progress = st.progress(0.0, text="Running benchmark...")

                for index, model_name in enumerate(selected_models):
                    progress.progress(index / len(selected_models), text=f"Processing {model_name}...")
                    try:
                        backend = make_backend(model_name, config)
                        fit_start = time.perf_counter()
                        backend.fit(X_train, y_train)
                        fit_time = time.perf_counter() - fit_start

                        predict_start = time.perf_counter()
                        pred = np.asarray(backend.predict(X_test))
                        probabilities = backend.predict_proba(X_test) if task == "Classification" else None
                        predict_time = time.perf_counter() - predict_start
                        predictions[model_name] = pred

                        if task == "Regression":
                            metrics = regression_metrics(y_test, pred)
                        else:
                            metrics = classification_metrics(
                                y_test,
                                pred,
                                probabilities=probabilities,
                                classes=backend.class_labels(),
                            )
                        row = {
                            "Model": model_name,
                            **metrics,
                            "Fit Time (s)": fit_time,
                            "Predict Time (s)": predict_time,
                            "Total Time (s)": fit_time + predict_time,
                        }
                        results.append(row)

                        if shap_payload is None and backend.supports_shap:
                            payload = backend.shap_payload(X_test)
                            if payload is not None:
                                shap_payload = (model_name, *payload)
                    except Exception as exc:
                        st.error(f"{model_name}: {exc}")

                progress.empty()
                if results:
                    results_df = pd.DataFrame(results)
                    st.session_state["benchmark_output"] = {
                        "results": results_df,
                        "predictions": predictions,
                        "y_test": np.asarray(y_test),
                        "X_test": X_test.copy(),
                        "target": target,
                        "task": task,
                    }

                    st.subheader("Integrated report")
                    st.dataframe(results_df.style.format(precision=4), use_container_width=True)
                    col_metric, col_time = st.columns(2)
                    with col_metric:
                        render_primary_metric_plot(results_df, task)
                    with col_time:
                        render_time_plot(results_df)

                    st.subheader("Detailed visual analysis")
                    if task == "Regression":
                        detail_fig = plot_regression(y_test, predictions, target)
                    else:
                        detail_fig = plot_classification(y_test, predictions, target)
                    st.pyplot(detail_fig)
                    plt.close(detail_fig)

                    if shap_payload is not None:
                        shap_name, shap_model, shap_X = shap_payload
                        st.subheader(f"Feature importance via SHAP: {shap_name}")
                        shap_fig = generate_shap_plot(shap_model, shap_X, f"SHAP summary: {shap_name}")
                        if shap_fig is not None:
                            st.pyplot(shap_fig)
                            plt.close(shap_fig)

                    result_df = X_test.copy()
                    result_df[f"Actual_{target}"] = np.asarray(y_test)
                    for model_name, pred in predictions.items():
                        result_df[f"Pred_{model_name}"] = pred
                    st.download_button(
                        "Download benchmark predictions",
                        data=csv_bytes(result_df),
                        file_name="tabtester_benchmark_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        stored = st.session_state.get("benchmark_output")
        if stored and stored.get("target") == target and stored.get("task") == task:
            with st.expander("Last benchmark result", expanded=False):
                st.dataframe(stored["results"].style.format(precision=4), use_container_width=True)

    with tab_predict:
        st.markdown("### Predict new rows")
        predict_model = st.selectbox("Prediction model", available_models, key="predict_model")
        new_file = st.file_uploader("Upload rows to predict (CSV)", type=["csv"], key="predict_file")
        if predict_model == "TabFM":
            st.warning(TABFM_LICENSE_NOTICE)
        if new_file is not None:
            try:
                new_df = read_csv(new_file)
                new_features = align_feature_columns(new_df, list(X.columns))
                st.dataframe(new_df.head(100), use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction CSV error: {exc}")
                new_features = None

            if new_features is not None and st.button("Train on all labeled rows and predict", type="primary"):
                if validate_tabfm_use([predict_model], tabfm_ack, tabfm_checkpoint_path):
                    try:
                        backend = make_backend(predict_model, config)
                        with st.spinner(f"Fitting {predict_model} and predicting..."):
                            backend.fit(X, y)
                            pred = np.asarray(backend.predict(new_features))
                        output = new_df.copy()
                        output[f"Pred_{target}_{predict_model}"] = pred
                        st.dataframe(output.head(100), use_container_width=True)
                        st.download_button(
                            "Download predictions",
                            data=csv_bytes(output),
                            file_name="tabtester_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    with tab_impute:
        st.markdown("### Missing target imputation")
        st.caption("Train on rows where the selected target is present, then predict only the missing target rows.")
        missing_mask = df[target].isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            st.info(f"The selected target column '{target}' has no missing values.")
        elif not available_foundation_models:
            st.warning("No foundation-model backend is installed for imputation.")
        else:
            st.write(f"Missing target rows: **{missing_count}**")
            impute_model = st.selectbox("Imputation model", available_foundation_models, key="impute_model")
            if impute_model == "TabFM":
                st.warning(TABFM_LICENSE_NOTICE)
            if st.button("Run imputation", type="primary"):
                if validate_tabfm_use([impute_model], tabfm_ack, tabfm_checkpoint_path):
                    try:
                        X_missing_raw = df.loc[missing_mask].drop(columns=[target] + excluded)
                        X_missing = align_feature_columns(X_missing_raw, list(X.columns))
                        backend = make_backend(impute_model, config)
                        with st.spinner(f"Imputing with {impute_model}..."):
                            backend.fit(X, y)
                            pred = np.asarray(backend.predict(X_missing))
                        output = df.copy()
                        imputed_column = f"{target}_imputed"
                        output[imputed_column] = output[target]
                        output.loc[missing_mask, imputed_column] = pred
                        st.dataframe(output.loc[missing_mask].head(100), use_container_width=True)
                        st.download_button(
                            "Download imputed dataset",
                            data=csv_bytes(output),
                            file_name=f"tabtester_imputed_{target}.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Imputation failed: {exc}")


if __name__ == "__main__":
    main()
