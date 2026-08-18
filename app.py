from __future__ import annotations

import hashlib
import importlib.metadata
import io
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
    registered_model_names,
)
from tabtester.plotting import configure_matplotlib_font, plot_classification, plot_regression
from tabtester.utils import (
    align_feature_columns,
    classification_metrics,
    complete_target_columns,
    csv_bytes,
    impute_with_backup,
    missing_column_summary,
    prepare_benchmark_target,
    read_csv_with_info,
    regression_metrics,
    safe_stratify,
)

APP_TITLE = "Tabtester"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
ENABLE_JAPANESE_SUPPORT = os.getenv("ENABLE_JAPANESE_SUPPORT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MATPLOTLIB_FONT = configure_matplotlib_font(ENABLE_JAPANESE_SUPPORT)

TABFM_LICENSE_NOTICE = (
    "TabFM default pretrained weights are licensed separately from the TabFM source code. "
    "They are restricted to non-commercial, non-production use."
)

CSV_ENCODING_LABELS = {
    "Auto": "auto",
    "UTF-8": "utf-8",
    "UTF-8 BOM": "utf-8-sig",
    "UTF-16": "utf-16",
}
CSV_DELIMITER_LABELS = {
    "Auto": "auto",
    "Comma": "comma",
    "Tab": "tab",
    "Semicolon": "semicolon",
}


def render_csv_input_options(prefix: str) -> tuple[str, str]:
    encodings = dict(CSV_ENCODING_LABELS)
    if ENABLE_JAPANESE_SUPPORT:
        encodings["CP932"] = "cp932"

    with st.expander("CSV input options", expanded=False):
        col_encoding, col_delimiter = st.columns(2)
        with col_encoding:
            encoding_label = st.selectbox(
                "Encoding",
                options=list(encodings),
                index=0,
                key=f"{prefix}_csv_encoding",
            )
        with col_delimiter:
            delimiter_label = st.selectbox(
                "Delimiter",
                options=list(CSV_DELIMITER_LABELS),
                index=0,
                key=f"{prefix}_csv_delimiter",
            )
    return encodings[encoding_label], CSV_DELIMITER_LABELS[delimiter_label]


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
    if ENABLE_JAPANESE_SUPPORT:
        st.text("Japanese support: enabled")
        st.text(f"Matplotlib font: {MATPLOTLIB_FONT or 'Japanese-capable font not found'}")
    offline = os.getenv("HF_HUB_OFFLINE", "0").strip() == "1"
    st.text(f"Foundation cache: {'offline' if offline else 'network fallback allowed'}")
    st.text(f"HF_HOME: {os.getenv('HF_HOME', 'default cache')}")
    st.text(f"Model cache root: {os.getenv('TABTESTER_MODEL_CACHE', '/models')}")
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


def figure_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def benchmark_overview(outputs: list[dict[str, Any]], task: str) -> pd.DataFrame:
    metric = "R2" if task == "Regression" else "Accuracy"
    rows: list[dict[str, Any]] = []
    for order, output in enumerate(outputs, start=1):
        results = output.get("results")
        best_model = ""
        best_score: float | None = None
        if isinstance(results, pd.DataFrame) and not results.empty and metric in results.columns:
            best_index = results[metric].astype(float).idxmax()
            best_model = str(results.loc[best_index, "Model"])
            best_score = float(results.loc[best_index, metric])

        model_errors = output.get("model_errors", {})
        error_text = output.get("error", "")
        if model_errors:
            model_error_text = "; ".join(f"{name}: {message}" for name, message in model_errors.items())
            error_text = "; ".join(part for part in [error_text, model_error_text] if part)

        rows.append(
            {
                "Order": order,
                "Target": output.get("target", ""),
                "Status": output.get("status", ""),
                "Best model": best_model,
                f"Best {metric}": best_score,
                "Models completed": output.get("completed_models", 0),
                "Models requested": output.get("requested_models", 0),
                "Errors": error_text,
            }
        )
    return pd.DataFrame(rows)


def safe_filename_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned or "target"


def render_benchmark_target_result(
    output: dict[str, Any],
    task: str,
    source_df: pd.DataFrame,
) -> None:
    target = str(output["target"])
    status = str(output.get("status", ""))
    st.markdown(f"### Target: {target}")
    st.caption(f"Status: {status}")

    results_df = output.get("results")
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        st.error(str(output.get("error") or "No model completed successfully for this target."))
        model_errors = output.get("model_errors", {})
        if model_errors:
            with st.expander("Model errors", expanded=True):
                for model_name, message in model_errors.items():
                    st.error(f"{model_name}: {message}")
        return

    st.subheader("Integrated report")
    st.dataframe(results_df.style.format(precision=4), use_container_width=True)
    col_metric, col_time = st.columns(2)
    with col_metric:
        render_primary_metric_plot(results_df, task)
    with col_time:
        render_time_plot(results_df)

    predictions = output["predictions"]
    y_test = output["y_test"]
    st.subheader("Detailed visual analysis")
    if task == "Regression":
        detail_fig = plot_regression(y_test, predictions, target)
    else:
        detail_fig = plot_classification(y_test, predictions, target)
    st.pyplot(detail_fig)
    plt.close(detail_fig)

    shap_image = output.get("shap_image")
    shap_name = output.get("shap_name")
    if shap_image is not None:
        st.subheader(f"Feature importance via SHAP: {shap_name}")
        st.image(io.BytesIO(shap_image), use_container_width=True)

    model_errors = output.get("model_errors", {})
    if model_errors:
        with st.expander("Model errors", expanded=False):
            for model_name, message in model_errors.items():
                st.error(f"{model_name}: {message}")

    result_df = source_df.loc[output["test_index"], output["feature_columns"]].copy()
    result_df[f"Actual_{target}"] = np.asarray(y_test)
    for model_name, pred in predictions.items():
        result_df[f"Pred_{model_name}"] = pred
    safe_target = safe_filename_component(target)
    st.download_button(
        "Download benchmark predictions",
        data=csv_bytes(result_df),
        file_name=f"tabtester_benchmark_predictions_{safe_target}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"benchmark_download::{target}",
    )


def set_benchmark_result_page(page: int) -> None:
    st.session_state["benchmark_result_page"] = page


def render_benchmark_results(
    outputs: list[dict[str, Any]],
    task: str,
    source_df: pd.DataFrame,
) -> None:
    if not outputs:
        return

    st.subheader("Benchmark overview")
    overview = benchmark_overview(outputs, task)
    score_column = "Best R2" if task == "Regression" else "Best Accuracy"
    formatters = {score_column: "{:.4f}"}
    st.dataframe(overview.style.format(formatters, na_rep=""), hide_index=True, use_container_width=True)

    page_options = list(range(len(outputs)))
    if st.session_state.get("benchmark_result_page") not in page_options:
        st.session_state["benchmark_result_page"] = 0
    page = st.selectbox(
        "Result page",
        page_options,
        format_func=lambda index: f"{index + 1} / {len(outputs)} - {outputs[index]['target']} [{outputs[index].get('status', '')}]",
        key="benchmark_result_page",
    )
    col_previous, col_next = st.columns(2)
    col_previous.button(
        "Previous result",
        disabled=page == 0,
        use_container_width=True,
        on_click=set_benchmark_result_page,
        args=(page - 1,),
        key="benchmark_previous_page",
    )
    col_next.button(
        "Next result",
        disabled=page == len(outputs) - 1,
        use_container_width=True,
        on_click=set_benchmark_result_page,
        args=(page + 1,),
        key="benchmark_next_page",
    )
    render_benchmark_target_result(outputs[page], task, source_df)


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
        all_models = registered_model_names()
        available_model_set = set(available_models)
        default_models = [name for name in ["TabICLv2", "XGBoost (Default)"] if name in available_model_set]
        if not default_models and available_models:
            default_models = available_models[:1]

        st.markdown("**Models to benchmark**")
        selected_models = []
        for model_name in all_models:
            toggle_key = f"benchmark_model_toggle::{model_name}"
            is_available = model_name in available_model_set
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = model_name in default_models
            if not is_available:
                st.session_state[toggle_key] = False
            enabled = st.toggle(
                model_name,
                key=toggle_key,
                disabled=not is_available,
                help=None if is_available else "Required backend is not installed in this environment.",
            )
            if enabled and is_available:
                selected_models.append(model_name)

        unavailable_models = [name for name in all_models if name not in available_model_set]
        if unavailable_models:
            st.caption("Unavailable backends are shown disabled: " + ", ".join(unavailable_models))

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

    train_encoding, train_delimiter = render_csv_input_options("train")
    train_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    if train_file is None:
        st.info("Upload a CSV dataset to begin.")
        return

    raw_train = train_file.getvalue()
    signature_payload = raw_train + f"\0{train_encoding}\0{train_delimiter}".encode("utf-8")
    dataset_signature = hashlib.sha256(signature_payload).hexdigest()
    try:
        df, used_encoding = read_csv_with_info(
            train_file,
            enable_japanese_support=ENABLE_JAPANESE_SUPPORT,
            encoding_mode=train_encoding,
            delimiter_mode=train_delimiter,
        )
        st.caption(
            f"CSV parser: encoding={used_encoding}, delimiter={train_delimiter}"
        )
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

    target_candidates = complete_target_columns(df)
    missing_targets = missing_column_summary(df)
    if not missing_targets.empty:
        st.warning(
            f"{len(missing_targets)} column(s) contain missing values and are excluded from Target columns."
        )
        st.dataframe(
            missing_targets.style.format({"Missing %": "{:.1f}%"}),
            hide_index=True,
            use_container_width=True,
        )

    if not target_candidates:
        st.error("No complete data column is available for Target columns.")
        return

    benchmark_targets = st.multiselect(
        "Target columns",
        target_candidates,
        default=[target_candidates[-1]],
        help=(
            "Select one or more complete target columns. Every selected target is removed "
            "from the feature set for every benchmark target."
        ),
    )
    if not benchmark_targets:
        st.error("Select at least one target column.")
        return

    candidate_excluded = [column for column in df.columns if column not in benchmark_targets]
    excluded = st.multiselect("Exclude columns", candidate_excluded, default=[])
    st.caption(
        "Leakage guard: all selected target columns are excluded from the benchmark feature set, "
        "regardless of which target is currently being evaluated."
    )

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
        st.caption(
            "Targets are processed sequentially in the selected order. Results are stored as each target "
            "finishes, and a failed target or model does not stop later targets."
        )
        st.caption(
            "Timing includes local model loading/preparation performed inside fit. "
            "When HF Hub is offline and checkpoints were prefetched into the persistent model cache, network download time is zero."
        )
        test_size = st.slider("Test fraction", 0.05, 0.5, DEFAULT_TEST_SIZE, 0.05)

        if "TabICLv2" in selected_models and len(df) < 300:
            st.warning(
                "TabICLv2 was pretrained on datasets starting around 300 rows; smaller datasets "
                "should be treated as an empirical test case."
            )
        if "TabFM" in selected_models and len(df) > 100:
            st.info(
                "TabFM uses a bounded context window and samples context rows when the training table "
                "is larger than its configured context size."
            )
        if task == "Classification" and "TabFM" in selected_models:
            oversized_targets = [
                target
                for target in benchmark_targets
                if df[target].nunique(dropna=False) > 10
            ]
            if oversized_targets:
                st.warning(
                    "TabFM v1.0.0 supports at most 10 classes and may fail for: "
                    + ", ".join(oversized_targets)
                )

        if st.button("Run benchmark", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("Select at least one model.")
            elif validate_tabfm_use(selected_models, tabfm_ack, tabfm_checkpoint_path):
                outputs: list[dict[str, Any]] = []
                total_steps = max(1, len(benchmark_targets) * len(selected_models))
                progress = st.progress(0.0, text="Running benchmark...")
                live_overview = st.empty()

                for target_index, target in enumerate(benchmark_targets):
                    target_output: dict[str, Any] = {
                        "target": target,
                        "task": task,
                        "status": "Running",
                        "requested_models": len(selected_models),
                        "completed_models": 0,
                        "model_errors": {},
                    }
                    try:
                        X_target, y_target = prepare_benchmark_target(
                            df,
                            target,
                            benchmark_targets,
                            excluded,
                            task,
                        )
                        stratify = safe_stratify(y_target, test_size) if task == "Classification" else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_target,
                            y_target,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=stratify,
                        )

                        results: list[dict[str, Any]] = []
                        predictions: dict[str, np.ndarray] = {}
                        model_errors: dict[str, str] = {}
                        shap_image: bytes | None = None
                        shap_name: str | None = None

                        for model_index, model_name in enumerate(selected_models):
                            step = target_index * len(selected_models) + model_index
                            progress.progress(
                                step / total_steps,
                                text=(
                                    f"Target {target_index + 1}/{len(benchmark_targets)}: {target} - "
                                    f"{model_name}"
                                ),
                            )
                            try:
                                backend = make_backend(model_name, config)
                                fit_start = time.perf_counter()
                                backend.fit(X_train, y_train)
                                fit_time = time.perf_counter() - fit_start

                                predict_start = time.perf_counter()
                                pred = np.asarray(backend.predict(X_test))
                                probabilities = (
                                    backend.predict_proba(X_test)
                                    if task == "Classification"
                                    else None
                                )
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
                                results.append(
                                    {
                                        "Model": model_name,
                                        **metrics,
                                        "Fit Time (s)": fit_time,
                                        "Predict Time (s)": predict_time,
                                        "Total Time (s)": fit_time + predict_time,
                                    }
                                )

                                if shap_image is None and backend.supports_shap:
                                    payload = backend.shap_payload(X_test)
                                    if payload is not None:
                                        shap_model, shap_X = payload
                                        shap_fig = generate_shap_plot(
                                            shap_model,
                                            shap_X,
                                            f"SHAP summary: {model_name} - {target}",
                                        )
                                        if shap_fig is not None:
                                            shap_image = figure_png_bytes(shap_fig)
                                            shap_name = model_name
                            except Exception as exc:
                                model_errors[model_name] = str(exc)

                        results_df = pd.DataFrame(results)
                        completed_models = len(results)
                        if completed_models == 0:
                            status = "Failed"
                            error = "All selected models failed for this target."
                        elif completed_models < len(selected_models):
                            status = "Partial"
                            error = ""
                        else:
                            status = "Done"
                            error = ""

                        target_output.update(
                            {
                                "status": status,
                                "error": error,
                                "results": results_df,
                                "predictions": predictions,
                                "y_test": np.asarray(y_test),
                                "test_index": X_test.index.tolist(),
                                "feature_columns": list(X_test.columns),
                                "completed_models": completed_models,
                                "model_errors": model_errors,
                                "shap_image": shap_image,
                                "shap_name": shap_name,
                            }
                        )
                    except Exception as exc:
                        target_output.update(
                            {
                                "status": "Failed",
                                "error": str(exc),
                                "results": pd.DataFrame(),
                            }
                        )

                    outputs.append(target_output)
                    st.session_state["benchmark_output"] = {
                        "dataset_signature": dataset_signature,
                        "targets": list(benchmark_targets),
                        "excluded": list(excluded),
                        "task": task,
                        "outputs": outputs.copy(),
                    }
                    live_overview.dataframe(
                        benchmark_overview(outputs, task),
                        hide_index=True,
                        use_container_width=True,
                    )
                    progress.progress(
                        min(1.0, ((target_index + 1) * len(selected_models)) / total_steps),
                        text=f"Completed target {target_index + 1}/{len(benchmark_targets)}: {target}",
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                progress.empty()
                live_overview.empty()

        stored = st.session_state.get("benchmark_output")
        if (
            stored
            and stored.get("dataset_signature") == dataset_signature
            and stored.get("targets") == list(benchmark_targets)
            and stored.get("excluded") == list(excluded)
            and stored.get("task") == task
        ):
            render_benchmark_results(stored.get("outputs", []), task, df)

    with tab_predict:
        st.markdown("### Predict new rows")
        if st.session_state.get("prediction_target") not in benchmark_targets:
            st.session_state["prediction_target"] = benchmark_targets[0]
        prediction_target = st.selectbox(
            "Prediction target",
            benchmark_targets,
            key="prediction_target",
            help="Prediction uses the same leakage-safe feature set as the benchmark.",
        )
        predict_model = st.selectbox("Prediction model", available_models, key="predict_model")
        if predict_model == "TabFM":
            st.warning(TABFM_LICENSE_NOTICE)

        try:
            X_predict, y_predict = prepare_benchmark_target(
                df,
                prediction_target,
                benchmark_targets,
                excluded,
                task,
            )
            prediction_setup_error = None
            if task == "Classification":
                st.caption(f"Classes: {y_predict.nunique(dropna=False)}")
        except Exception as exc:
            X_predict = None
            y_predict = None
            prediction_setup_error = str(exc)
            st.error(f"Prediction target setup failed: {prediction_setup_error}")

        predict_encoding, predict_delimiter = render_csv_input_options("predict")
        new_file = st.file_uploader("Upload rows to predict (CSV)", type=["csv"], key="predict_file")
        if new_file is not None and X_predict is not None:
            try:
                new_df, used_prediction_encoding = read_csv_with_info(
                    new_file,
                    enable_japanese_support=ENABLE_JAPANESE_SUPPORT,
                    encoding_mode=predict_encoding,
                    delimiter_mode=predict_delimiter,
                )
                new_features = align_feature_columns(new_df, list(X_predict.columns))
                st.caption(
                    f"Prediction CSV parser: encoding={used_prediction_encoding}, "
                    f"delimiter={predict_delimiter}"
                )
                st.dataframe(new_df.head(100), use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction CSV error: {exc}")
                new_features = None

            if new_features is not None and st.button(
                "Train on all labeled rows and predict",
                type="primary",
            ):
                if validate_tabfm_use([predict_model], tabfm_ack, tabfm_checkpoint_path):
                    try:
                        backend = make_backend(predict_model, config)
                        with st.spinner(f"Fitting {predict_model} and predicting..."):
                            backend.fit(X_predict, y_predict)
                            pred = np.asarray(backend.predict(new_features))
                        output = new_df.copy()
                        output[f"Pred_{prediction_target}_{predict_model}"] = pred
                        st.dataframe(output.head(100), use_container_width=True)
                        st.download_button(
                            "Download predictions",
                            data=csv_bytes(output),
                            file_name=(
                                f"tabtester_predictions_{safe_filename_component(prediction_target)}.csv"
                            ),
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    with tab_impute:
        st.markdown("### Missing target imputation")
        st.caption(
            "Benchmark targets must be complete. Select a column with missing values here to impute it separately."
        )
        impute_target_options = [
            str(column)
            for column in df.columns
            if bool(df[column].isna().any())
        ]
        if not impute_target_options:
            st.info("No column with missing target values is available for imputation.")
        elif not available_foundation_models:
            st.warning("No foundation-model backend is installed for imputation.")
        else:
            impute_target = st.selectbox(
                "Imputation target",
                impute_target_options,
                format_func=lambda column: f"{column} ({int(df[column].isna().sum())} missing)",
                key="impute_target",
            )
            missing_mask = df[impute_target].isna()
            missing_count = int(missing_mask.sum())
            st.write(f"Missing target rows: **{missing_count}**")
            impute_model = st.selectbox("Imputation model", available_foundation_models, key="impute_model")
            if impute_model == "TabFM":
                st.warning(TABFM_LICENSE_NOTICE)
            if st.button("Run imputation", type="primary"):
                if validate_tabfm_use([impute_model], tabfm_ack, tabfm_checkpoint_path):
                    try:
                        impute_excluded = [column for column in excluded if column != impute_target]
                        labeled_mask = ~missing_mask
                        X_impute_raw = df.loc[labeled_mask].drop(
                            columns=[impute_target] + impute_excluded
                        )
                        y_impute = df.loc[labeled_mask, impute_target].copy()
                        if X_impute_raw.shape[1] == 0:
                            raise ValueError("No feature columns remain for imputation.")

                        if task == "Regression":
                            y_impute_numeric = pd.to_numeric(y_impute, errors="coerce")
                            if y_impute_numeric.isna().any():
                                raise ValueError("Regression imputation target contains non-numeric values.")
                            y_impute = y_impute_numeric
                        elif y_impute.nunique(dropna=False) < 2:
                            raise ValueError("Classification imputation requires at least two target classes.")

                        X_missing_raw = df.loc[missing_mask].drop(
                            columns=[impute_target] + impute_excluded
                        )
                        X_missing = align_feature_columns(
                            X_missing_raw,
                            list(X_impute_raw.columns),
                        )
                        backend = make_backend(impute_model, config)
                        with st.spinner(f"Imputing with {impute_model}..."):
                            backend.fit(X_impute_raw, y_impute)
                            pred = np.asarray(backend.predict(X_missing))
                        output, backup_column = impute_with_backup(
                            df,
                            impute_target,
                            missing_mask,
                            pred,
                        )
                        st.success(
                            f"Filled missing values in '{impute_target}'. "
                            f"Original values are preserved in '{backup_column}'."
                        )
                        st.dataframe(output.loc[missing_mask].head(100), use_container_width=True)
                        st.download_button(
                            "Download imputed dataset",
                            data=csv_bytes(output),
                            file_name=f"tabtester_imputed_{impute_target}.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Imputation failed: {exc}")



if __name__ == "__main__":
    main()
