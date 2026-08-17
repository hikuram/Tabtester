from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BackendConfig:
    task: str
    device: str = "auto"
    random_state: int = 42
    n_trials: int = 10
    time_budget: int = 30
    tabicl_n_estimators: int = 8
    tabicl_batch_size: int = 4
    tabicl_kv_cache: bool | str = False
    tabicl_use_amp: bool | str = "auto"
    tabicl_offload_mode: bool | str = "auto"
    tabfm_checkpoint_path: str | None = None


class ModelBackend(ABC):
    name: str
    family: str = "other"
    supports_probability: bool = False
    supports_shap: bool = False

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model: Any = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelBackend":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame):
        return None

    def shap_payload(self, X: pd.DataFrame):
        return None

    def class_labels(self):
        return getattr(self.model, "classes_", None)
