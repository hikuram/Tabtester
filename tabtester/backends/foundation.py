from __future__ import annotations

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
        checkpoint_path = self.config.tabfm_checkpoint_path or None
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


class KumoTabularBackend(ModelBackend):
    name = "KumoTabular"
    family = "foundation"
    supports_probability = True

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._device = "cpu"
        self._feature_stypes: dict[str, object] = {}
        self._class_lookup: dict[str, object] = {}
        self._sorted_classes: list[object] = []
        self._class_labels: np.ndarray | None = None
        self._last_query_id: int | None = None
        self._last_predictions: np.ndarray | None = None
        self._last_probabilities: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KumoTabularBackend":
        import sdm

        if self.config.kumo_size not in {"small", "medium", "large"}:
            raise ValueError("KumoTabular size must be one of: small, medium, large.")
        if self.config.task == "Classification" and y.nunique(dropna=False) > 10:
            raise ValueError("KumoTabular supports at most 10 classes.")

        self._device = resolve_device(self.config.device)
        self._feature_stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        unsupported = [
            name
            for name, stype in self._feature_stypes.items()
            if str(stype) not in {"numerical", "categorical"}
        ]
        if unsupported:
            raise ValueError(
                "KumoTabular currently accepts numeric and categorical feature columns in Tabtester. "
                "Exclude or encode unsupported columns: " + ", ".join(map(str, unsupported))
            )

        x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self._feature_stypes,
            device=self._device,
        )
        target_name = str(y.name) if y.name is not None else "__target__"
        target_stype = "numerical" if self.config.task == "Regression" else "categorical"
        y_context = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: target_stype},
            device=self._device,
        )

        task = "regression" if self.config.task == "Regression" else "classification"
        self.model = sdm.models.KumoTabular(
            task=task,
            size=self.config.kumo_size,
            device=self._device,
        )
        generator = torch.Generator(device=self._device).manual_seed(self.config.random_state)
        device_type = torch.device(self._device).type
        with torch.amp.autocast(
            device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                num_estimators=self.config.kumo_n_estimators,
                estimator_batch_size=1,
                generator=generator,
            )

        if self.config.task == "Classification":
            values = list(pd.unique(y.dropna()))
            self._class_lookup = {str(value): value for value in values}
            try:
                self._sorted_classes = list(pd.Series(values).sort_values(kind="stable"))
            except (TypeError, ValueError):
                self._sorted_classes = values
        else:
            self._class_lookup = {}
            self._sorted_classes = []

        self._class_labels = None
        self._last_query_id = None
        self._last_predictions = None
        self._last_probabilities = None
        return self

    def _predict_output(self, X: pd.DataFrame):
        import sdm

        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self._feature_stypes,
            device=self._device,
        )
        device_type = torch.device(self._device).type
        with torch.amp.autocast(
            device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            return self.model.predict(x=x_query)

    def _predict_classification(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        query_id = id(X)
        if (
            self._last_query_id == query_id
            and self._last_predictions is not None
            and self._last_probabilities is not None
        ):
            return self._last_predictions, self._last_probabilities

        output = self._predict_output(X).cpu().to_pandas()
        probabilities = output.to_numpy(dtype=float)
        output_columns = [str(column) for column in output.columns]
        labels = [self._class_lookup.get(column) for column in output_columns]
        if any(label is None for label in labels):
            if len(self._sorted_classes) != len(output_columns):
                raise RuntimeError(
                    "KumoTabular returned a class layout that could not be aligned to the training labels."
                )
            labels = list(self._sorted_classes)
        class_labels = np.asarray(labels)
        predictions = class_labels[np.argmax(probabilities, axis=1)]

        self._class_labels = class_labels
        self._last_query_id = query_id
        self._last_predictions = np.asarray(predictions)
        self._last_probabilities = probabilities
        return self._last_predictions, self._last_probabilities

    def predict(self, X: pd.DataFrame):
        if self.config.task == "Classification":
            predictions, _ = self._predict_classification(X)
            return predictions
        output = self._predict_output(X)
        return output.numerical.float().mean(dim=-1).cpu().numpy()

    def predict_proba(self, X: pd.DataFrame):
        if self.config.task != "Classification":
            return None
        _, probabilities = self._predict_classification(X)
        return probabilities

    def class_labels(self):
        if self.config.task != "Classification":
            return None
        return self._class_labels
