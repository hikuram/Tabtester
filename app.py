from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
import time
from typing import Any, Mapping, Sequence

import altair as alt
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
from tabtester.plotting import plot_classification, plot_regression
from tabtester.recommendation import (
    GOAL_TYPES,
    SearchPlan,
    generate_candidates,
    normalize_design_space,
    score_candidates,
    select_shortlist,
    suggest_sample_count,
    top_fit_score,
    validate_design_space,
    validate_target_spec,
)
from tabtester.utils import (
    align_feature_columns,
    arrow_safe_dataframe,
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
TABFM_LICENSE_NOTICE = (
    "TabFM default pretrained weights are licensed separately from the TabFM source code. "
    "They are restricted to non-commercial, non-production use."
)
RECOMMENDATION_TABFM_NOTICE = (
    "TabFM is intentionally excluded from Recommendation because the default pretrained weights "
    "cannot be used for commercial decision-making. It remains available in Benchmark."
)

CSV_ENCODING_LABELS = {
    "Auto": "auto",
    "UTF-8": "utf-8",
    "UTF-8 BOM": "utf-8-sig",
    "UTF-16": "utf-16",
    "CP932": "cp932",
}
CSV_DELIMITER_LABELS = {
    "Auto": "auto",
    "Comma": "comma",
    "Tab": "tab",
    "Semicolon": "semicolon",
}


def render_csv_input_options(prefix: str) -> tuple[str, str]:
    encodings = CSV_ENCODING_LABELS

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


def recommendation_signature(
    data_key: str,
    properties: Sequence[str],
    target_spec: pd.DataFrame,
    design_space: pd.DataFrame,
    mixture_variables: Sequence[str],
    mixture_total: float,
    selected_models: Sequence[str],
) -> str:
    payload = "|".join(
        [
            data_key,
            ",".join(properties),
            target_spec.to_json(orient="split", double_precision=12),
            design_space.to_json(orient="split", double_precision=12),
            ",".join(mixture_variables),
            f"{mixture_total:.12g}",
            ",".join(selected_models),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def format_elapsed(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.2f} s"
    minutes, sec = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minute = divmod(int(minutes), 60)
    return f"{hours}h {minute}m {sec:.0f}s"


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
    st.dataframe(results_df.style.format(precision=4), width="stretch")
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
        st.image(io.BytesIO(shap_image), width="stretch")

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
        width="stretch",
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
    st.dataframe(overview.style.format(formatters, na_rep=""), hide_index=True, width="stretch")

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
        width="stretch",
        on_click=set_benchmark_result_page,
        args=(page - 1,),
        key="benchmark_previous_page",
    )
    col_next.button(
        "Next result",
        disabled=page == len(outputs) - 1,
        width="stretch",
        on_click=set_benchmark_result_page,
        args=(page + 1,),
        key="benchmark_next_page",
    )
    render_benchmark_target_result(outputs[page], task, source_df)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() >= max(2, int(df[column].notna().sum() * 0.9)):
            columns.append(column)
    return columns


def _reference_value(series: pd.Series) -> object:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.9)):
        return float(numeric.median())
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else ""


def _target_defaults(df: pd.DataFrame, properties: Sequence[str]) -> pd.DataFrame:
    rows = []
    for property_name in properties:
        values = pd.to_numeric(df[property_name], errors="coerce").dropna()
        rows.append(
            {
                "Property": property_name,
                "Goal": "Range",
                "Lower": float(values.min()) if not values.empty else np.nan,
                "Target": float(values.median()) if not values.empty else np.nan,
                "Upper": float(values.max()) if not values.empty else np.nan,
                "Priority": "Medium",
                "Hard": False,
            }
        )
    return pd.DataFrame(rows)


def _design_defaults(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    rows = []
    for variable in variables:
        values = pd.to_numeric(df[variable], errors="coerce").dropna()
        observed_min = float(values.min()) if not values.empty else np.nan
        observed_max = float(values.max()) if not values.empty else np.nan
        rows.append(
            {
                "Variable": variable,
                "Observed min": observed_min,
                "Observed max": observed_max,
                "Search min": observed_min,
                "Search max": observed_max,
                "Step": np.nan,
                "Active": True,
            }
        )
    return pd.DataFrame(rows)


@st.dialog("Mixture constraint", width="large")
def edit_mixture_constraint(active_variables: list[str]) -> None:
    existing = st.session_state.get("recommend_mixture", {})
    existing_variables = [name for name in existing.get("variables", []) if name in active_variables]
    selected = st.multiselect(
        "Variables whose sum is fixed",
        active_variables,
        default=existing_variables,
        help="Example: Resin A + Resin B + Filler = 100 wt%.",
    )
    total = st.number_input(
        "Fixed total",
        value=float(existing.get("total", 100.0)),
        step=1.0,
    )
    with st.container(horizontal=True):
        if st.button("Save", type="primary", icon=":material/save:"):
            if selected and len(selected) < 2:
                st.error("Select at least two variables, or clear the constraint.")
            else:
                st.session_state["recommend_mixture"] = {
                    "variables": selected,
                    "total": float(total),
                }
                st.rerun()
        if st.button("Clear", icon=":material/delete:"):
            st.session_state["recommend_mixture"] = {"variables": [], "total": 100.0}
            st.rerun()


def _nearest_existing_rows(
    candidate: pd.Series,
    df: pd.DataFrame,
    design_space: pd.DataFrame,
    columns_to_show: Sequence[str],
    k: int = 3,
) -> pd.DataFrame:
    table = normalize_design_space(design_space)
    active = table[table["Active"]]
    variables = active["Variable"].tolist()
    if not variables:
        return pd.DataFrame()
    training = df[variables].apply(pd.to_numeric, errors="coerce")
    valid = training.notna().all(axis=1)
    if not valid.any():
        return pd.DataFrame()
    mins = active.set_index("Variable")["Observed min"].reindex(variables).to_numpy(dtype=float)
    maxs = active.set_index("Variable")["Observed max"].reindex(variables).to_numpy(dtype=float)
    spans = np.where((maxs - mins) > 1e-12, maxs - mins, 1.0)
    train_scaled = (training.loc[valid].to_numpy(dtype=float) - mins) / spans
    candidate_scaled = (candidate[variables].to_numpy(dtype=float) - mins) / spans
    distances = np.linalg.norm(train_scaled - candidate_scaled, axis=1)
    order = np.argsort(distances)[: min(k, len(distances))]
    original_indices = training.loc[valid].index.to_numpy()[order]
    available_columns = [column for column in columns_to_show if column in df.columns]
    result = df.loc[original_indices, available_columns].copy()
    result.insert(0, "Distance", distances[order])
    return result


@st.dialog("Candidate details", width="large")
def show_candidate_details(
    candidate_id: str,
    output: Mapping[str, Any],
    source_df: pd.DataFrame,
) -> None:
    results = output["results"]
    matches = results.index[results["Candidate"] == candidate_id].tolist()
    if not matches:
        st.error("Candidate not found.")
        return
    row_index = matches[0]
    candidate = results.loc[row_index]
    design_variables = output["design_variables"]
    properties = output["properties"]
    models = output["models"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target fit", f"{candidate['Target fit']:.1f}%")
    col2.metric("Hard violations", int(candidate["Hard violations"]))
    col3.metric("Domain", str(candidate["Domain"]))
    col4.metric("Model disagreement", f"{candidate['Model disagreement']:.3f}")

    st.markdown("### Proposed conditions")
    input_table = pd.DataFrame(
        {"Variable": design_variables, "Value": [candidate[name] for name in design_variables]}
    )
    st.dataframe(arrow_safe_dataframe(input_table), hide_index=True)

    st.markdown("### Predicted properties")
    property_rows = []
    prediction_detail = output["prediction_detail"]
    for property_name in properties:
        item: dict[str, Any] = {
            "Property": property_name,
            "Consensus": candidate[f"Pred {property_name}"],
        }
        for model_name in models:
            values = prediction_detail.get(model_name, {}).get(property_name)
            item[model_name] = values[row_index] if values is not None else np.nan
        property_rows.append(item)
    st.dataframe(arrow_safe_dataframe(pd.DataFrame(property_rows)), hide_index=True)
    st.caption(
        "Consensus is the median prediction across the selected models. Model disagreement is the "
        "mean normalized median absolute deviation across target properties."
    )

    st.markdown("### Nearest existing experiments")
    columns_to_show = list(dict.fromkeys(design_variables + properties))
    nearest = _nearest_existing_rows(
        candidate,
        source_df,
        output["design_space"],
        columns_to_show,
        k=3,
    )
    if nearest.empty:
        st.caption("No complete existing rows were available for distance comparison.")
    else:
        st.dataframe(arrow_safe_dataframe(nearest), hide_index=True)


def _goal_rules(chart: alt.Chart, spec: pd.Series, axis: str) -> alt.Chart:
    goal = str(spec["Goal"])
    lower = pd.to_numeric(pd.Series([spec["Lower"]]), errors="coerce").iloc[0]
    target = pd.to_numeric(pd.Series([spec["Target"]]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([spec["Upper"]]), errors="coerce").iloc[0]
    layers: list[alt.Chart] = [chart]
    if axis == "x":
        if np.isfinite(lower):
            layers.append(alt.Chart(pd.DataFrame({"value": [lower]})).mark_rule(strokeDash=[4, 4]).encode(x="value:Q"))
        if np.isfinite(upper):
            layers.append(alt.Chart(pd.DataFrame({"value": [upper]})).mark_rule(strokeDash=[4, 4]).encode(x="value:Q"))
        if goal == "Close to" and np.isfinite(target):
            layers.append(alt.Chart(pd.DataFrame({"value": [target]})).mark_rule().encode(x="value:Q"))
    else:
        if np.isfinite(lower):
            layers.append(alt.Chart(pd.DataFrame({"value": [lower]})).mark_rule(strokeDash=[4, 4]).encode(y="value:Q"))
        if np.isfinite(upper):
            layers.append(alt.Chart(pd.DataFrame({"value": [upper]})).mark_rule(strokeDash=[4, 4]).encode(y="value:Q"))
        if goal == "Close to" and np.isfinite(target):
            layers.append(alt.Chart(pd.DataFrame({"value": [target]})).mark_rule().encode(y="value:Q"))
    combined = layers[0]
    for layer in layers[1:]:
        combined = combined + layer
    return combined


def render_recommendation_results(output: Mapping[str, Any], source_df: pd.DataFrame) -> None:
    results: pd.DataFrame = output["results"]
    shortlist: pd.DataFrame = output["shortlist"]
    properties: list[str] = output["properties"]
    target_spec: pd.DataFrame = output["target_spec"]

    timing = output.get("timing", {})
    total_seconds = float(timing.get("total_seconds", 0.0))

    st.markdown("## :material/science: Candidate recommendations")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates evaluated", f"{len(results):,}")
    c2.metric("Pareto candidates", int(results["Pareto"].sum()))
    c3.metric("Hard-feasible", int((results["Hard violations"] == 0).sum()))
    c4.metric("Total time", format_elapsed(total_seconds) if total_seconds > 0 else "-")
    st.caption(
        f"Search mode: {output['search_mode']} · Requested plan: {output['plan_text']} · "
        f"Actual search stopped after {output['search_rounds']} round(s) · Models: {len(output['models'])}."
    )

    if timing:
        with st.expander("Execution timing", icon=":material/timer:"):
            phase_rows = []
            phase_labels = [
                ("Training", "training_seconds"),
                ("Candidate generation", "candidate_generation_seconds"),
                ("Prediction", "prediction_seconds"),
                ("Scoring and Pareto", "scoring_seconds"),
                ("Shortlist", "shortlist_seconds"),
                ("Other", "other_seconds"),
            ]
            for label, key in phase_labels:
                seconds = float(timing.get(key, 0.0))
                phase_rows.append(
                    {
                        "Phase": label,
                        "Time (s)": seconds,
                        "Share (%)": (100.0 * seconds / total_seconds) if total_seconds > 0 else 0.0,
                    }
                )
            phase_df = pd.DataFrame(phase_rows)
            st.dataframe(
                phase_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Time (s)": st.column_config.NumberColumn(format="%.3f"),
                    "Share (%)": st.column_config.ProgressColumn(
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f%%",
                    ),
                },
            )

            detail_rows = timing.get("model_property", [])
            if detail_rows:
                st.caption("Fit time is measured once per model/property. Predict time is accumulated across search rounds.")
                st.dataframe(
                    arrow_safe_dataframe(pd.DataFrame(detail_rows)),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Fit time (s)": st.column_config.NumberColumn(format="%.3f"),
                        "Predict time (s)": st.column_config.NumberColumn(format="%.3f"),
                    },
                )

            prediction_seconds = float(timing.get("prediction_seconds", 0.0))
            prediction_values = int(timing.get("prediction_values", 0))
            timing_cols = st.columns(3)
            timing_cols[0].metric("Training", format_elapsed(float(timing.get("training_seconds", 0.0))))
            timing_cols[1].metric("Prediction", format_elapsed(prediction_seconds))
            if prediction_seconds > 0 and prediction_values > 0:
                timing_cols[2].metric("Prediction throughput", f"{prediction_values / prediction_seconds:,.0f} values/s")
            else:
                timing_cols[2].metric("Prediction throughput", "-")

    if int((results["Hard violations"] == 0).sum()) == 0 and bool(target_spec["Hard"].any()):
        st.warning(
            "No candidates satisfy all hard targets. The shortlist shows the closest trade-offs instead.",
            icon=":material/warning:",
        )

    chart_col, summary_col = st.columns([2, 1])
    with chart_col:
        st.markdown("### Pareto trade-off")
        if len(properties) >= 2:
            selector_cols = st.columns(2)
            x_property = selector_cols[0].selectbox(
                "X axis", properties, index=0, key="recommend_x_property"
            )
            y_property = selector_cols[1].selectbox(
                "Y axis", properties, index=1 if len(properties) > 1 else 0, key="recommend_y_property"
            )
            x_column = f"Pred {x_property}"
            y_column = f"Pred {y_property}"
        else:
            x_property = properties[0]
            y_property = None
            x_column = f"Pred {x_property}"
            y_column = "Model disagreement"

        if len(results) > 5000:
            pareto_rows = results[results["Pareto"]]
            remaining = results[~results["Pareto"]]
            sample_n = max(0, 5000 - len(pareto_rows))
            chart_data = pd.concat(
                [pareto_rows, remaining.sample(min(sample_n, len(remaining)), random_state=42)],
                ignore_index=True,
            )
        else:
            chart_data = results

        base = (
            alt.Chart(chart_data)
            .mark_circle(opacity=0.55)
            .encode(
                x=alt.X(f"{x_column}:Q", title=x_property),
                y=alt.Y(f"{y_column}:Q", title=y_property or "Model disagreement"),
                color=alt.Color("Domain:N", title="Domain"),
                shape=alt.Shape("Pareto:N", title="Pareto"),
                size=alt.Size("Model disagreement:Q", title="Disagreement", scale=alt.Scale(range=[30, 240])),
                tooltip=[
                    "Candidate:N",
                    alt.Tooltip("Target fit:Q", format=".1f"),
                    "Hard violations:Q",
                    "Domain:N",
                    alt.Tooltip("Model disagreement:Q", format=".3f"),
                    alt.Tooltip(f"{x_column}:Q", title=x_property, format=".4g"),
                    alt.Tooltip(f"{y_column}:Q", title=y_property or "Model disagreement", format=".4g"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        x_spec = target_spec[target_spec["Property"] == x_property].iloc[0]
        chart = _goal_rules(base, x_spec, "x")
        if y_property is not None:
            y_spec = target_spec[target_spec["Property"] == y_property].iloc[0]
            chart = _goal_rules(chart, y_spec, "y")
        st.altair_chart(chart)
        st.caption("Dashed rules show entered lower/upper targets; a solid rule shows a Close-to target.")

    with summary_col:
        st.markdown("### First shortlist candidate")
        if shortlist.empty:
            st.caption("No shortlist is available.")
        else:
            first = shortlist.iloc[0]
            st.metric("Candidate", first["Candidate"])
            st.metric("Target fit", f"{first['Target fit']:.1f}%")
            st.caption(f"Domain: **{first['Domain']}**")
            st.caption(f"Model disagreement: **{first['Model disagreement']:.3f}**")
            st.caption(f"Nearest existing distance: **{first['Nearest distance']:.3f}**")
            for property_name in properties:
                st.metric(property_name, f"{first[f'Pred {property_name}']:.5g}")

    view = st.segmented_control(
        "Candidate view",
        ["Shortlist", "Pareto", "All candidates"],
        default="Shortlist",
    )
    view = view or "Shortlist"
    if view == "Shortlist":
        table = shortlist.copy()
    elif view == "Pareto":
        table = results[results["Pareto"]].copy()
    else:
        table = results.copy()

    primary_columns = [
        "Candidate",
        "Pareto",
        "Target fit",
        "Hard violations",
        "Domain",
        "Model disagreement",
        "Nearest distance",
    ]
    prediction_columns = [f"Pred {name}" for name in properties]
    design_columns = output["design_variables"]
    display_columns = [column for column in primary_columns + prediction_columns + design_columns if column in table.columns]
    table = table.loc[:, display_columns]
    selection = st.dataframe(
        arrow_safe_dataframe(table),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"recommend_results_{view}",
        column_config={
            "Target fit": st.column_config.ProgressColumn(
                "Target fit",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            ),
            "Model disagreement": st.column_config.NumberColumn(format="%.3f"),
            "Nearest distance": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    if selection.selection.rows:
        selected_index = selection.selection.rows[0]
        selected_id = str(table.iloc[selected_index]["Candidate"])
        show_candidate_details(selected_id, output, source_df)

    st.download_button(
        "Download candidate table",
        data=csv_bytes(results),
        file_name="tabtester_recommendations.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_recommendation_page(
    df: pd.DataFrame,
    data_key: str,
    available_models: list[str],
    backend_config: BackendConfig,
    random_state: int,
    default_properties: Sequence[str] | None = None,
    protected_targets: Sequence[str] | None = None,
) -> None:
    st.markdown("## :material/experiment: Recommend")
    st.caption(
        "Define target properties and a design space. Tabtester screens candidate conditions with multiple models and returns data-grounded trade-offs rather than a single claimed optimum."
    )

    if "TabFM" in available_models:
        st.info(RECOMMENDATION_TABFM_NOTICE, icon=":material/policy:")

    numeric_columns = _numeric_columns(df)
    if not numeric_columns:
        st.error("Recommendation requires numeric target properties and numeric design variables.")
        return

    st.markdown("### 1. Target properties")
    preferred_properties = [name for name in (default_properties or []) if name in numeric_columns]
    default_property_list = preferred_properties or [numeric_columns[-1]]
    properties = st.multiselect(
        "Properties to optimize",
        numeric_columns,
        default=default_property_list,
        key="recommend_properties",
    )
    if not properties:
        st.info("Select at least one target property.")
        return

    target_editor_key = "recommend_target_" + hashlib.sha1("|".join(properties).encode()).hexdigest()[:8]
    target_spec = st.data_editor(
        _target_defaults(df, properties),
        key=target_editor_key,
        hide_index=True,
        disabled=["Property"],
        column_config={
            "Goal": st.column_config.SelectboxColumn("Goal", options=list(GOAL_TYPES), required=True),
            "Priority": st.column_config.SelectboxColumn(
                "Priority", options=["Low", "Medium", "High"], required=True
            ),
            "Hard": st.column_config.CheckboxColumn("Hard constraint"),
            "Lower": st.column_config.NumberColumn("Lower", format="%.6g"),
            "Target": st.column_config.NumberColumn("Target", format="%.6g"),
            "Upper": st.column_config.NumberColumn("Upper", format="%.6g"),
        },
    )
    st.caption(
        "For Close to, Target is the preferred value. If it is also a hard constraint, enter Lower and Upper as the acceptable band."
    )

    st.markdown("### 2. Design space")
    excluded_options = [column for column in df.columns if column not in properties]
    excluded = st.multiselect(
        "Exclude non-model columns",
        excluded_options,
        default=[],
        key="recommend_excluded",
        help="Exclude IDs, timestamps, notes, or other columns that should not be model inputs.",
    )
    protected = set(protected_targets or [])
    feature_columns = [
        column
        for column in df.columns
        if column not in set(properties)
        and column not in set(excluded)
        and column not in protected
    ]
    protected_not_properties = [name for name in protected if name not in set(properties) and name in df.columns]
    if protected_not_properties:
        st.caption(
            "Leakage guard: selected Benchmark target columns are excluded from Recommendation model inputs: "
            + ", ".join(sorted(protected_not_properties))
        )
    numeric_features = [column for column in feature_columns if column in numeric_columns]
    if not numeric_features:
        st.error("No numeric design variables remain after selecting properties/exclusions.")
        return

    default_design = numeric_features[: min(8, len(numeric_features))]
    design_variables = st.multiselect(
        "Design variables",
        numeric_features,
        default=default_design,
        key="recommend_design_variables",
        help="Selected variables can vary. Other model inputs are fixed at their median or most common observed value.",
    )
    if not design_variables:
        st.info("Select at least one design variable.")
        return

    design_editor_key = "recommend_design_" + hashlib.sha1("|".join(design_variables).encode()).hexdigest()[:8]
    design_space = st.data_editor(
        _design_defaults(df, design_variables),
        key=design_editor_key,
        hide_index=True,
        disabled=["Variable", "Observed min", "Observed max"],
        column_config={
            "Observed min": st.column_config.NumberColumn(format="%.6g"),
            "Observed max": st.column_config.NumberColumn(format="%.6g"),
            "Search min": st.column_config.NumberColumn(format="%.6g", required=True),
            "Search max": st.column_config.NumberColumn(format="%.6g", required=True),
            "Step": st.column_config.NumberColumn(
                "Step",
                format="%.6g",
                help="Leave blank for continuous space-filling sampling.",
            ),
            "Active": st.column_config.CheckboxColumn("Use"),
        },
    )
    active_variables = normalize_design_space(design_space)
    active_variables = active_variables[active_variables["Active"]]["Variable"].tolist()

    mixture = st.session_state.setdefault("recommend_mixture", {"variables": [], "total": 100.0})
    mixture["variables"] = [name for name in mixture.get("variables", []) if name in active_variables]
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Configure mixture constraint", icon=":material/functions:"):
            edit_mixture_constraint(active_variables)
        if mixture["variables"]:
            st.badge(
                f"{' + '.join(mixture['variables'])} = {float(mixture['total']):g}",
                color="blue",
                icon=":material/check_circle:",
            )
        else:
            st.caption("No mixture-sum constraint")

    fixed_columns = [column for column in feature_columns if column not in active_variables]
    fixed_values = {column: _reference_value(df[column]) for column in fixed_columns}
    with st.expander("Fixed model inputs", icon=":material/push_pin:"):
        if fixed_values:
            st.caption("These inputs are held at the median (numeric) or most common observed value (categorical).")
            st.dataframe(
                arrow_safe_dataframe(
                    pd.DataFrame(
                        {"Feature": list(fixed_values), "Fixed value": list(fixed_values.values())}
                    )
                ),
                hide_index=True,
            )
        else:
            st.caption("All model inputs are active design variables.")

    extrapolated = []
    normalized_space = normalize_design_space(design_space)
    for row in normalized_space[normalized_space["Active"]].itertuples(index=False):
        if float(getattr(row, "_3")) < float(getattr(row, "_1")) or float(getattr(row, "_4")) > float(getattr(row, "_2")):
            extrapolated.append(row.Variable)
    if extrapolated:
        st.warning(
            f"Search range extends beyond observed data for: {', '.join(extrapolated)}. Candidates there will be marked Extrapolation.",
            icon=":material/warning:",
        )

    st.markdown("### 3. Search")
    effort = st.segmented_control(
        "Search effort", ["Quick", "Balanced", "Thorough"], default="Balanced"
    )
    effort = effort or "Balanced"
    sample_mode = st.segmented_control("Sampling budget", ["Auto", "Manual"], default="Auto")
    sample_mode = sample_mode or "Auto"

    mixture_variables = mixture.get("variables", [])
    mixture_total = float(mixture.get("total", 100.0))
    try:
        plan = suggest_sample_count(
            design_space,
            effort=effort,
            mixture_variables=mixture_variables,
        )
    except Exception as exc:
        st.error(f"Could not estimate search size: {exc}")
        return

    if sample_mode == "Manual" and plan.mode != "exhaustive":
        manual_count = int(
            st.number_input(
                "Candidate samples",
                min_value=128,
                max_value=131_072,
                value=min(max(plan.initial_count, 128), 131_072),
                step=128,
            )
        )
        plan = SearchPlan(
            mode="sample",
            initial_count=manual_count,
            max_count=manual_count,
            effective_dimensions=plan.effective_dimensions,
            exhaustive_count=plan.exhaustive_count,
            reason="Manual candidate count.",
        )

    plan_cols = st.columns(3)
    plan_cols[0].metric(
        "Recommended candidates" if plan.mode != "exhaustive" else "Full grid",
        f"{plan.initial_count:,}",
    )
    plan_cols[1].metric("Effective dimensions", plan.effective_dimensions)
    plan_cols[2].metric("Search mode", "Full grid" if plan.mode == "exhaustive" else "Space filling")
    st.caption(plan.reason)
    auto_expand = st.toggle(
        "Auto-expand while the top shortlist is still improving",
        value=(sample_mode == "Auto" and plan.mode != "exhaustive"),
        disabled=(sample_mode != "Auto" or plan.mode == "exhaustive"),
    )
    if auto_expand:
        st.caption(f"Search may expand up to {plan.max_count:,} candidates and stops when the top-fit score stabilizes.")

    recommendation_models = [name for name in available_models if name != "TabFM"]
    preferred = [
        name
        for name in ["TabICLv2", "XGBoost (Default)", "CatBoost"]
        if name in recommendation_models
    ]
    if not preferred and recommendation_models:
        preferred = recommendation_models[: min(2, len(recommendation_models))]
    selected_models = st.multiselect(
        "Models contributing to consensus",
        recommendation_models,
        default=preferred,
        key="recommend_models",
    )
    st.caption(
        "Consensus uses the median across selected models. Disagreement is reported separately instead of being hidden inside the score."
    )
    current_config_key = recommendation_signature(
        data_key,
        properties,
        target_spec,
        design_space,
        mixture_variables,
        mixture_total,
        selected_models,
    )

    if st.button(
        "Generate candidate recommendations",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=not selected_models,
    ):
        errors = validate_target_spec(target_spec)
        errors.extend(validate_design_space(design_space, mixture_variables, mixture_total))
        if not selected_models:
            errors.append("Select at least one recommendation model.")
        if errors:
            for error in errors:
                st.error(error)
            return

        observed_targets = df[properties].apply(pd.to_numeric, errors="coerce")
        model_feature_columns = feature_columns
        fixed_for_generation = {
            column: fixed_values.get(column, _reference_value(df[column])) for column in model_feature_columns
        }
        for variable in active_variables:
            fixed_for_generation.setdefault(variable, _reference_value(df[variable]))

        config = backend_config
        trained: dict[str, dict[str, Any]] = {}
        overall_start = time.perf_counter()
        timing_totals = {
            "training_seconds": 0.0,
            "candidate_generation_seconds": 0.0,
            "prediction_seconds": 0.0,
            "scoring_seconds": 0.0,
            "shortlist_seconds": 0.0,
            "prediction_values": 0,
        }
        model_property_timing = {
            (model_name, property_name): {"fit": 0.0, "predict": 0.0}
            for model_name in selected_models
            for property_name in properties
        }
        status = st.status("Training surrogate models...", expanded=True)
        try:
            for model_name in selected_models:
                trained[model_name] = {}
                for property_name in properties:
                    y_values = pd.to_numeric(df[property_name], errors="coerce")
                    mask = y_values.notna()
                    if int(mask.sum()) < 3:
                        raise ValueError(f"{property_name}: at least 3 labeled rows are required.")
                    X_train = df.loc[mask, model_feature_columns].copy()
                    y_train = y_values.loc[mask]
                    status.write(f"Fitting {model_name} → {property_name} ({len(y_train)} rows)")
                    backend = make_backend(model_name, config)
                    fit_start = time.perf_counter()
                    backend.fit(X_train, y_train)
                    fit_seconds = time.perf_counter() - fit_start
                    timing_totals["training_seconds"] += fit_seconds
                    model_property_timing[(model_name, property_name)]["fit"] += fit_seconds
                    trained[model_name][property_name] = backend

            all_candidates: list[pd.DataFrame] = []
            accumulated_predictions: dict[str, dict[str, list[np.ndarray]]] = {
                model_name: {property_name: [] for property_name in properties}
                for model_name in selected_models
            }
            previous_fit: float | None = None
            scored = None
            search_mode = ""
            rounds = 0
            total_requested = plan.initial_count if plan.mode != "exhaustive" else plan.initial_count
            while True:
                rounds += 1
                if plan.mode == "exhaustive":
                    batch_count = plan.initial_count
                else:
                    already = sum(len(item) for item in all_candidates)
                    remaining = max(0, plan.max_count - already)
                    batch_count = min(plan.initial_count, remaining)
                    if batch_count <= 0:
                        break

                status.write(f"Generating candidate batch {rounds} ({batch_count:,} requested)")
                generation_start = time.perf_counter()
                candidates, search_mode = generate_candidates(
                    design_space,
                    fixed_for_generation,
                    batch_count,
                    random_state=int(random_state) + rounds - 1,
                    mixture_variables=mixture_variables,
                    mixture_total=mixture_total,
                )
                timing_totals["candidate_generation_seconds"] += time.perf_counter() - generation_start
                candidates = candidates.loc[:, model_feature_columns]
                if all_candidates:
                    previous = pd.concat(all_candidates, ignore_index=True)
                    merged = pd.concat([previous, candidates], ignore_index=True).drop_duplicates()
                    candidates = merged.iloc[len(previous) :].copy()
                if candidates.empty:
                    break
                all_candidates.append(candidates.reset_index(drop=True))

                for model_name in selected_models:
                    for property_name in properties:
                        status.write(f"Predicting {model_name} → {property_name} on {len(candidates):,} candidates")
                        predict_start = time.perf_counter()
                        pred = np.asarray(trained[model_name][property_name].predict(candidates), dtype=float)
                        predict_seconds = time.perf_counter() - predict_start
                        timing_totals["prediction_seconds"] += predict_seconds
                        timing_totals["prediction_values"] += int(len(pred))
                        model_property_timing[(model_name, property_name)]["predict"] += predict_seconds
                        accumulated_predictions[model_name][property_name].append(pred)

                combined_candidates = pd.concat(all_candidates, ignore_index=True)
                combined_predictions = {
                    model_name: {
                        property_name: np.concatenate(accumulated_predictions[model_name][property_name])
                        for property_name in properties
                    }
                    for model_name in selected_models
                }
                scoring_start = time.perf_counter()
                scored = score_candidates(
                    combined_candidates,
                    combined_predictions,
                    target_spec,
                    observed_targets,
                    design_space,
                    df[active_variables],
                )
                timing_totals["scoring_seconds"] += time.perf_counter() - scoring_start
                current_fit = top_fit_score(scored.results)
                status.write(
                    f"Current top-shortlist mean fit: {current_fit:.2f}% after {len(combined_candidates):,} unique candidates "
                    f"· elapsed {format_elapsed(time.perf_counter() - overall_start)}"
                )

                if plan.mode == "exhaustive" or not auto_expand:
                    break
                if previous_fit is not None and abs(current_fit - previous_fit) < 0.25:
                    status.write("Top-shortlist score stabilized; stopping automatic expansion.")
                    break
                previous_fit = current_fit
                if len(combined_candidates) >= plan.max_count:
                    break
                total_requested += plan.initial_count

            if scored is None:
                raise ValueError("No feasible candidates were generated.")

            shortlist_start = time.perf_counter()
            shortlist = select_shortlist(
                scored.results,
                active_variables,
                limit=10,
                min_separation=0.06,
            )
            timing_totals["shortlist_seconds"] += time.perf_counter() - shortlist_start
            total_seconds = time.perf_counter() - overall_start
            measured_seconds = sum(
                float(timing_totals[key])
                for key in [
                    "training_seconds",
                    "candidate_generation_seconds",
                    "prediction_seconds",
                    "scoring_seconds",
                    "shortlist_seconds",
                ]
            )
            timing_output = {
                **timing_totals,
                "total_seconds": total_seconds,
                "other_seconds": max(0.0, total_seconds - measured_seconds),
                "model_property": [
                    {
                        "Model": model_name,
                        "Property": property_name,
                        "Fit time (s)": values["fit"],
                        "Predict time (s)": values["predict"],
                    }
                    for (model_name, property_name), values in model_property_timing.items()
                ],
            }
            output = {
                "data_key": data_key,
                "config_key": current_config_key,
                "results": scored.results,
                "shortlist": shortlist,
                "prediction_detail": scored.prediction_detail,
                "target_spec": target_spec.copy(),
                "design_space": design_space.copy(),
                "design_variables": active_variables,
                "properties": list(properties),
                "models": list(selected_models),
                "search_mode": search_mode,
                "search_rounds": rounds,
                "timing": timing_output,
                "plan_text": (
                    f"full grid {plan.initial_count:,}"
                    if plan.mode == "exhaustive"
                    else f"{plan.initial_count:,} initial / {plan.max_count:,} max"
                ),
            }
            st.session_state["recommendation_output"] = output
            status.update(
                label=f"Recommendation search complete ({format_elapsed(total_seconds)})",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            failed_seconds = time.perf_counter() - overall_start
            status.update(
                label=f"Recommendation search failed after {format_elapsed(failed_seconds)}",
                state="error",
                expanded=True,
            )
            st.error(str(exc))

    output = st.session_state.get("recommendation_output")
    if output and output.get("data_key") == data_key:
        if output.get("config_key") == current_config_key:
            render_recommendation_results(output, df)
        else:
            st.info(
                "Recommendation settings changed since the last run. Generate recommendations again to refresh the result.",
                icon=":material/info:",
            )



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
                value="",
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
    st.dataframe(arrow_safe_dataframe(df.head(100)), width="stretch")

    target_candidates = complete_target_columns(df)
    missing_targets = missing_column_summary(df)
    if not missing_targets.empty:
        st.warning(
            f"{len(missing_targets)} column(s) contain missing values and are excluded from Target columns."
        )
        st.dataframe(
            missing_targets.style.format({"Missing %": "{:.1f}%"}),
            hide_index=True,
            width="stretch",
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
    recommendation_config = build_config(
        "Regression",
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

    tab_eval, tab_predict, tab_impute, tab_recommend = st.tabs(
        [
            "Benchmark and Evaluation",
            "Predict New Rows",
            "Impute Missing Target",
            "Recommend Candidates",
        ]
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

        if st.button("Run benchmark", type="primary", width="stretch"):
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
                        width="stretch",
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
                            encoding_mode=predict_encoding,
                    delimiter_mode=predict_delimiter,
                )
                new_features = align_feature_columns(new_df, list(X_predict.columns))
                st.caption(
                    f"Prediction CSV parser: encoding={used_prediction_encoding}, "
                    f"delimiter={predict_delimiter}"
                )
                st.dataframe(arrow_safe_dataframe(new_df.head(100)), width="stretch")
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
                        st.dataframe(arrow_safe_dataframe(output.head(100)), width="stretch")
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
                        st.dataframe(arrow_safe_dataframe(output.loc[missing_mask].head(100)), width="stretch")
                        st.download_button(
                            "Download imputed dataset",
                            data=csv_bytes(output),
                            file_name=f"tabtester_imputed_{impute_target}.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Imputation failed: {exc}")


    with tab_recommend:
        render_recommendation_page(
            df,
            dataset_signature,
            available_models,
            recommendation_config,
            random_state,
            default_properties=benchmark_targets,
            protected_targets=benchmark_targets,
        )



if __name__ == "__main__":
    main()
