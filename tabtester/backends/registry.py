from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .base import BackendConfig, ModelBackend
from .foundation import TabFMBackend, TabICLv2Backend
from .gaussian_process import GenericMaternGPBackend
from .traditional import (
    AutoGluonBackend,
    CatBoostBackend,
    FLAMLBackend,
    LightGBMBackend,
    XGBoostDefaultBackend,
    XGBoostTunedBackend,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    dependencies: tuple[str, ...]
    backend_class: type[ModelBackend]
    tasks: tuple[str, ...] = ("Regression", "Classification")


MODEL_SPECS = (
    ModelSpec("TabFM", "foundation", ("tabfm",), TabFMBackend),
    ModelSpec("TabICLv2", "foundation", ("tabicl",), TabICLv2Backend),
    ModelSpec("XGBoost (Default)", "traditional", ("xgboost",), XGBoostDefaultBackend),
    ModelSpec("LightGBM", "traditional", ("lightgbm",), LightGBMBackend),
    ModelSpec("CatBoost", "traditional", ("catboost",), CatBoostBackend),
    ModelSpec(
        "GP (Generic Matern)",
        "gaussian_process",
        ("torch",),
        GenericMaternGPBackend,
        ("Regression",),
    ),
    ModelSpec("XGBoost (Tuned)", "automl", ("xgboost", "optuna"), XGBoostTunedBackend),
    ModelSpec("FLAML", "automl", ("flaml",), FLAMLBackend),
    ModelSpec("AutoGluon", "automl", ("autogluon.tabular",), AutoGluonBackend),
)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def registered_model_names() -> list[str]:
    """Return all model backends exposed by the application."""
    return [spec.name for spec in MODEL_SPECS]


def model_supports_task(name: str, task: str) -> bool:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return task in spec.tasks
    raise KeyError(f"Unknown model backend: {name}")


def available_model_names(task: str | None = None) -> list[str]:
    return [
        spec.name
        for spec in MODEL_SPECS
        if all(_module_available(module) for module in spec.dependencies)
        and (task is None or task in spec.tasks)
    ]


def foundation_model_names(task: str | None = None) -> list[str]:
    available = set(available_model_names(task))
    return [spec.name for spec in MODEL_SPECS if spec.family == "foundation" and spec.name in available]


def make_backend(name: str, config: BackendConfig) -> ModelBackend:
    for spec in MODEL_SPECS:
        if spec.name == name:
            if config.task not in spec.tasks:
                raise ValueError(f"{name} does not support {config.task}.")
            missing = [module for module in spec.dependencies if not _module_available(module)]
            if missing:
                raise ImportError(f"Missing dependencies for {name}: {missing}")
            return spec.backend_class(config)
    raise KeyError(f"Unknown model backend: {name}")
