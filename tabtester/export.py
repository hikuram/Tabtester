from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import plot_classification, plot_regression


def safe_filename_component(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value)
    )
    return cleaned or "item"


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def figure_png_bytes(fig: Any) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def primary_metric_figure(results: pd.DataFrame, task: str):
    metric = "R2" if task == "Regression" else "Accuracy"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results["Model"], results[metric])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def execution_time_figure(results: pd.DataFrame):
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
    return fig


def benchmark_predictions_frame(output: Mapping[str, Any], source_df: pd.DataFrame) -> pd.DataFrame:
    target = str(output["target"])
    test_index = list(output.get("test_index", []))
    feature_columns = list(output.get("feature_columns", []))
    if not test_index:
        return pd.DataFrame()

    result_df = source_df.loc[test_index, feature_columns].copy()
    index_column = "Source index"
    while index_column in result_df.columns:
        index_column = "_" + index_column
    result_df.insert(0, index_column, test_index)
    result_df[f"Actual_{target}"] = np.asarray(output.get("y_test", []))
    for model_name, pred in output.get("predictions", {}).items():
        result_df[f"Pred_{model_name}"] = np.asarray(pred)
    return result_df


def benchmark_summary_frame(outputs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_columns: list[str] = []

    for output in outputs:
        results = output.get("results")
        if isinstance(results, pd.DataFrame):
            for column in results.columns:
                if column != "Model" and column not in metric_columns:
                    metric_columns.append(column)

    for output in outputs:
        target = str(output.get("target", ""))
        target_status = str(output.get("status", ""))
        results = output.get("results")
        completed_models: set[str] = set()

        if isinstance(results, pd.DataFrame) and not results.empty:
            for _, result_row in results.iterrows():
                model_name = str(result_row.get("Model", ""))
                completed_models.add(model_name)
                row: dict[str, Any] = {
                    "Target": target,
                    "Target status": target_status,
                    "Model": model_name,
                    "Model status": "Done",
                    "Error": "",
                }
                for column in metric_columns:
                    row[column] = result_row.get(column, np.nan)
                rows.append(row)

        for model_name, message in output.get("model_errors", {}).items():
            if model_name in completed_models:
                continue
            row = {
                "Target": target,
                "Target status": target_status,
                "Model": str(model_name),
                "Model status": "Failed",
                "Error": str(message),
            }
            for column in metric_columns:
                row[column] = np.nan
            rows.append(row)

        if not rows or not any(row["Target"] == target for row in rows):
            row = {
                "Target": target,
                "Target status": target_status,
                "Model": "",
                "Model status": "Failed",
                "Error": str(output.get("error", "")),
            }
            for column in metric_columns:
                row[column] = np.nan
            rows.append(row)

    base_columns = ["Target", "Target status", "Model", "Model status", "Error"]
    return pd.DataFrame(rows, columns=base_columns + metric_columns)


def failures_frame(outputs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for output in outputs:
        target = str(output.get("target", ""))
        target_error = str(output.get("error", "")).strip()
        if target_error:
            rows.append({"Target": target, "Model": "", "Error": target_error})
        for model_name, message in output.get("model_errors", {}).items():
            rows.append({"Target": target, "Model": str(model_name), "Error": str(message)})
    return pd.DataFrame(rows, columns=["Target", "Model", "Error"])


def run_settings_frame(sections: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for section, values in sections.items():
        for setting, value in values.items():
            if isinstance(value, (list, tuple, set)):
                rendered = " | ".join(str(item) for item in value)
            elif value is None:
                rendered = ""
            else:
                rendered = str(value)
            rows.append(
                {
                    "Section": str(section),
                    "Setting": str(setting),
                    "Value": rendered,
                }
            )
    return pd.DataFrame(rows, columns=["Section", "Setting", "Value"])


def build_benchmark_zip(
    outputs: Sequence[Mapping[str, Any]],
    task: str,
    source_df: pd.DataFrame,
    run_settings: pd.DataFrame,
) -> bytes:
    archive_buffer = io.BytesIO()
    export_errors: list[dict[str, str]] = []

    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("benchmark_summary.csv", dataframe_csv_bytes(benchmark_summary_frame(outputs)))
        archive.writestr("run_settings.csv", dataframe_csv_bytes(run_settings))
        archive.writestr("failures.csv", dataframe_csv_bytes(failures_frame(outputs)))

        for order, output in enumerate(outputs, start=1):
            target = str(output.get("target", f"target_{order}"))
            target_dir = f"{order:02d}_{safe_filename_component(target)}"
            results = output.get("results")

            if isinstance(results, pd.DataFrame) and not results.empty:
                archive.writestr(f"{target_dir}/metrics.csv", dataframe_csv_bytes(results))

                try:
                    predictions_df = benchmark_predictions_frame(output, source_df)
                    archive.writestr(
                        f"{target_dir}/predictions.csv",
                        dataframe_csv_bytes(predictions_df),
                    )
                except Exception as exc:
                    export_errors.append(
                        {"Target": target, "Artifact": "predictions.csv", "Error": str(exc)}
                    )

                try:
                    archive.writestr(
                        f"{target_dir}/metric_comparison.png",
                        figure_png_bytes(primary_metric_figure(results, task)),
                    )
                except Exception as exc:
                    export_errors.append(
                        {"Target": target, "Artifact": "metric_comparison.png", "Error": str(exc)}
                    )

                try:
                    archive.writestr(
                        f"{target_dir}/execution_time.png",
                        figure_png_bytes(execution_time_figure(results)),
                    )
                except Exception as exc:
                    export_errors.append(
                        {"Target": target, "Artifact": "execution_time.png", "Error": str(exc)}
                    )

                predictions = output.get("predictions", {})
                y_test = output.get("y_test")
                if predictions and y_test is not None:
                    detail_name = (
                        "actual_vs_predicted.png"
                        if task == "Regression"
                        else "confusion_matrix.png"
                    )
                    try:
                        if task == "Regression":
                            detail_fig = plot_regression(y_test, predictions, target)
                        else:
                            detail_fig = plot_classification(y_test, predictions, target)
                        archive.writestr(
                            f"{target_dir}/{detail_name}",
                            figure_png_bytes(detail_fig),
                        )
                    except Exception as exc:
                        export_errors.append(
                            {"Target": target, "Artifact": detail_name, "Error": str(exc)}
                        )

                shap_image = output.get("shap_image")
                if shap_image is not None:
                    shap_name = safe_filename_component(str(output.get("shap_name") or "model"))
                    archive.writestr(
                        f"{target_dir}/shap_{shap_name}.png",
                        bytes(shap_image),
                    )

            model_errors = output.get("model_errors", {})
            target_error = str(output.get("error", "")).strip()
            if model_errors or target_error:
                rows: list[dict[str, str]] = []
                if target_error:
                    rows.append({"Model": "", "Error": target_error})
                rows.extend(
                    {"Model": str(model_name), "Error": str(message)}
                    for model_name, message in model_errors.items()
                )
                archive.writestr(
                    f"{target_dir}/errors.csv",
                    dataframe_csv_bytes(pd.DataFrame(rows, columns=["Model", "Error"])),
                )

        if export_errors:
            archive.writestr(
                "export_errors.csv",
                dataframe_csv_bytes(
                    pd.DataFrame(export_errors, columns=["Target", "Artifact", "Error"])
                ),
            )

    return archive_buffer.getvalue()
