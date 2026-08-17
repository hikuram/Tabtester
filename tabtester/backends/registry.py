from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .base import BackendConfig, ModelBackend
from .foundation import TabFMBackend, TabICLv2Backend
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


MODEL_SPECS = (
    ModelSpec("TabFM", "foundation", ("tabfm",), TabFMBackend),
    ModelSpec("TabICLv2", "foundation", ("tabicl",), TabICLv2Backend),
    ModelSpec("XGBoost (Default)", "traditional", ("xgboost",), XGBoostDefaultBackend),
    ModelSpec("LightGBM", "traditional", ("lightgbm",), LightGBMBackend),
    ModelSpec("CatBoost", "traditional", ("catboost",), CatBoostBackend),
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


def available_model_names() -> list[str]:
    return [
        spec.name
        for spec in MODEL_SPECS
        if all(_module_available(module) for module in spec.dependencies)
    ]


def foundation_model_names() -> list[str]:
    available = set(available_model_names())
    return [spec.name for spec in MODEL_SPECS if spec.family == "foundation" and spec.name in available]


def make_backend(name: str, config: BackendConfig) -> ModelBackend:
    for spec in MODEL_SPECS:
        if spec.name == name:
            missing = [module for module in spec.dependencies if not _module_available(module)]
            if missing:
                raise ImportError(f"Missing dependencies for {name}: {missing}")
            return spec.backend_class(config)
    raise KeyError(f"Unknown model backend: {name}")
