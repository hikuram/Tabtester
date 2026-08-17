from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

from .base import BackendConfig, ModelBackend


class DummyFrameEncoder:
    def __init__(self) -> None:
        self.columns_: list[str] = []

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.get_dummies(X, dummy_na=True)
        self.columns_ = list(encoded.columns)
        return encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.get_dummies(X, dummy_na=True)
        return encoded.reindex(columns=self.columns_, fill_value=0)


class EncodedBackend(ModelBackend):
    supports_shap = True

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.encoder = DummyFrameEncoder()
        self.label_encoder: LabelEncoder | None = None

    def _encode_target(self, y: pd.Series):
        if self.config.task == "Regression":
            return np.asarray(y)
        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(np.asarray(y))

    def _decode_target(self, values):
        values_arr = np.asarray(values)
        if self.label_encoder is None:
            return values_arr
        return self.label_encoder.inverse_transform(values_arr.astype(int))

    def predict(self, X: pd.DataFrame):
        encoded = self.encoder.transform(X)
        return self._decode_target(self.model.predict(encoded))

    def predict_proba(self, X: pd.DataFrame):
        if self.config.task != "Classification" or not hasattr(self.model, "predict_proba"):
            return None
        return self.model.predict_proba(self.encoder.transform(X))

    def shap_payload(self, X: pd.DataFrame):
        if not self.supports_shap:
            return None
        return self.model, self.encoder.transform(X)

    def class_labels(self):
        if self.label_encoder is None:
            return None
        return self.label_encoder.classes_


class XGBoostDefaultBackend(EncodedBackend):
    name = "XGBoost (Default)"
    family = "traditional"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostDefaultBackend":
        import xgboost as xgb

        X_encoded = self.encoder.fit_transform(X)
        y_encoded = self._encode_target(y)
        if self.config.task == "Regression":
            self.model = xgb.XGBRegressor(random_state=self.config.random_state, n_jobs=-1)
        else:
            self.model = xgb.XGBClassifier(
                random_state=self.config.random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )
        self.model.fit(X_encoded, y_encoded)
        return self


class LightGBMBackend(EncodedBackend):
    name = "LightGBM"
    family = "traditional"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMBackend":
        import lightgbm as lgb

        X_encoded = self.encoder.fit_transform(X)
        y_encoded = self._encode_target(y)
        if self.config.task == "Regression":
            self.model = lgb.LGBMRegressor(random_state=self.config.random_state, verbose=-1, n_jobs=-1)
        else:
            self.model = lgb.LGBMClassifier(random_state=self.config.random_state, verbose=-1, n_jobs=-1)
        self.model.fit(X_encoded, y_encoded)
        return self


class CatBoostBackend(EncodedBackend):
    name = "CatBoost"
    family = "traditional"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostBackend":
        import catboost as cb

        X_encoded = self.encoder.fit_transform(X)
        y_encoded = self._encode_target(y)
        if self.config.task == "Regression":
            self.model = cb.CatBoostRegressor(
                random_state=self.config.random_state,
                verbose=0,
                thread_count=-1,
            )
        else:
            self.model = cb.CatBoostClassifier(
                random_state=self.config.random_state,
                verbose=0,
                thread_count=-1,
            )
        self.model.fit(X_encoded, y_encoded)
        return self


class XGBoostTunedBackend(EncodedBackend):
    name = "XGBoost (Tuned)"
    family = "automl"

    def _best_params(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        import optuna
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        is_regression = self.config.task == "Regression"

        if is_regression:
            n_splits = min(3, max(2, len(X) // 5))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)
        else:
            counts = np.bincount(y.astype(int))
            counts = counts[counts > 0]
            n_splits = min(3, int(counts.min())) if counts.size else 0
            if n_splits < 2:
                raise ValueError("Not enough samples per class for tuned XGBoost cross-validation.")
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.config.random_state,
                "n_jobs": -1,
            }
            scores = []
            split_iter = splitter.split(X, y) if not is_regression else splitter.split(X)
            for train_idx, valid_idx in split_iter:
                if is_regression:
                    estimator = xgb.XGBRegressor(**params)
                else:
                    estimator = xgb.XGBClassifier(**params, eval_metric="logloss")
                estimator.fit(X.iloc[train_idx], y[train_idx])
                pred = estimator.predict(X.iloc[valid_idx])
                if is_regression:
                    scores.append(r2_score(y[valid_idx], pred))
                else:
                    scores.append(accuracy_score(y[valid_idx], pred))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials)
        params = dict(study.best_params)
        params.update({"random_state": self.config.random_state, "n_jobs": -1})
        if not is_regression:
            params["eval_metric"] = "logloss"
        return params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostTunedBackend":
        import xgboost as xgb

        X_encoded = self.encoder.fit_transform(X)
        y_encoded = self._encode_target(y)
        params = self._best_params(X_encoded, y_encoded)
        if self.config.task == "Regression":
            self.model = xgb.XGBRegressor(**params)
        else:
            self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_encoded, y_encoded)
        return self


class FLAMLBackend(EncodedBackend):
    name = "FLAML"
    family = "automl"
    supports_shap = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FLAMLBackend":
        from flaml import AutoML

        X_encoded = self.encoder.fit_transform(X)
        y_encoded = self._encode_target(y)
        automl = AutoML()
        automl.fit(
            X_train=X_encoded,
            y_train=y_encoded,
            task="regression" if self.config.task == "Regression" else "classification",
            metric="r2" if self.config.task == "Regression" else "accuracy",
            time_budget=self.config.time_budget,
            verbose=0,
        )
        self.model = automl
        return self


class AutoGluonBackend(ModelBackend):
    name = "AutoGluon"
    family = "automl"
    supports_probability = True

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._path: Path | None = None
        self._target_name = "__tabtester_target__"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoGluonBackend":
        from autogluon.tabular import TabularPredictor

        train_data = X.copy()
        train_data[self._target_name] = np.asarray(y)
        self._path = Path(tempfile.mkdtemp(prefix="tabtester_autogluon_"))
        self.model = TabularPredictor(
            label=self._target_name,
            verbosity=0,
            path=str(self._path),
        ).fit(train_data, time_limit=self.config.time_budget, presets="medium_quality")
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X).to_numpy()

    def predict_proba(self, X: pd.DataFrame):
        if self.config.task != "Classification":
            return None
        values = self.model.predict_proba(X)
        return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)

    def class_labels(self):
        return getattr(self.model, "class_labels", None)
