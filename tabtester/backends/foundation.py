from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from .base import BackendConfig, ModelBackend


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False.")
    return device


def _tabfm_dtype(device: str):
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


@lru_cache(maxsize=8)
def _load_tabfm_base(task: str, device: str, checkpoint_path: str | None):
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    model_type = "regression" if task == "Regression" else "classification"
    return tabfm_v1_0_0.load(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=_tabfm_dtype(device),
        use_cache=True,
    )


class TabFMBackend(ModelBackend):
    name = "TabFM"
    family = "foundation"
    supports_probability = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabFMBackend":
        from tabfm import TabFMClassifier, TabFMRegressor

        if self.config.task == "Classification" and y.nunique(dropna=False) > 10:
            raise ValueError("TabFM v1.0.0 supports at most 10 classes.")
        device = resolve_device(self.config.device)
        checkpoint_path = self.config.tabfm_checkpoint_path or os.getenv("TABFM_CHECKPOINT_PATH") or None
        base_model = _load_tabfm_base(self.config.task, device, checkpoint_path)
        if self.config.task == "Regression":
            self.model = TabFMRegressor(model=base_model)
        else:
            self.model = TabFMClassifier(model=base_model)
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.config.task != "Classification":
            return None
        return self.model.predict_proba(X)


class TabICLv2Backend(ModelBackend):
    name = "TabICLv2"
    family = "foundation"
    supports_probability = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabICLv2Backend":
        from tabicl import TabICLClassifier, TabICLRegressor

        kwargs = {
            "n_estimators": self.config.tabicl_n_estimators,
            "batch_size": self.config.tabicl_batch_size,
            "kv_cache": self.config.tabicl_kv_cache,
            "device": resolve_device(self.config.device),
            "use_amp": self.config.tabicl_use_amp,
            "offload_mode": self.config.tabicl_offload_mode,
            "random_state": self.config.random_state,
        }
        if self.config.task == "Regression":
            self.model = TabICLRegressor(**kwargs)
        else:
            self.model = TabICLClassifier(**kwargs)
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if self.config.task != "Classification":
            return None
        return self.model.predict_proba(X)
