from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .base import BackendConfig, ModelBackend


def _matern52(x1: torch.Tensor, x2: torch.Tensor, lengthscales: torch.Tensor) -> torch.Tensor:
    diff = (x1[:, None, :] - x2[None, :, :]) / lengthscales
    radius = torch.sqrt(torch.sum(diff * diff, dim=-1).clamp_min(1e-30))
    root5 = math.sqrt(5.0)
    return (1.0 + root5 * radius + (5.0 / 3.0) * radius.square()) * torch.exp(
        -root5 * radius
    )


def _positive(raw: torch.Tensor, lower: float = 1e-3, upper: float = 1e3) -> torch.Tensor:
    return torch.exp(raw).clamp(lower, upper)


@dataclass
class _GenericPreprocessor:
    columns: list[str] | None = None
    medians: pd.Series | None = None
    lower: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit_transform(self, X: pd.DataFrame) -> torch.Tensor:
        encoded = pd.get_dummies(X, dummy_na=True)
        self.columns = list(encoded.columns)
        encoded = encoded.apply(pd.to_numeric, errors="coerce")
        self.medians = encoded.median(numeric_only=True)
        encoded = encoded.fillna(self.medians).fillna(0.0)
        values = encoded.to_numpy(dtype=float)
        self.lower = values.min(axis=0)
        upper = values.max(axis=0)
        self.scale = upper - self.lower
        self.scale[self.scale < 1e-12] = 1.0
        return torch.as_tensor((values - self.lower) / self.scale, dtype=torch.float64)

    def transform(self, X: pd.DataFrame) -> torch.Tensor:
        if self.columns is None or self.medians is None or self.lower is None or self.scale is None:
            raise RuntimeError("Generic GP preprocessor has not been fitted.")
        encoded = pd.get_dummies(X, dummy_na=True).reindex(columns=self.columns, fill_value=0)
        encoded = encoded.apply(pd.to_numeric, errors="coerce")
        encoded = encoded.fillna(self.medians).fillna(0.0)
        values = encoded.to_numpy(dtype=float)
        return torch.as_tensor((values - self.lower) / self.scale, dtype=torch.float64)


class _ExactTabularGP(torch.nn.Module):
    """Exact ARD Matern-5/2 GP for tabular regression."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.log_noise = torch.nn.Parameter(torch.tensor(math.log(0.05), dtype=torch.float64))
        self.log_lengthscales = torch.nn.Parameter(torch.zeros(input_dim, dtype=torch.float64))
        self.log_outputscale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return _positive(self.log_outputscale, 1e-4, 1e4) * _matern52(
            x1,
            x2,
            _positive(self.log_lengthscales),
        )

    def covariance(self, X: torch.Tensor) -> torch.Tensor:
        noise = _positive(self.log_noise, 1e-6, 1.0)
        eye = torch.eye(len(X), dtype=X.dtype, device=X.device)
        return self.kernel(X, X) + (noise + 1e-6) * eye


def _fit_exact_gp(
    model: _ExactTabularGP,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    max_iter: int,
    lr: float,
    mll_weight: float,
) -> float:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=1e-6,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def losses() -> tuple[torch.Tensor, torch.Tensor]:
        covariance = model.covariance(X)
        chol = torch.linalg.cholesky(covariance)
        alpha = torch.cholesky_solve(y[:, None], chol).squeeze(-1)
        inverse_diag = torch.cholesky_inverse(chol).diagonal().clamp_min(1e-12)
        loo_variance = 1.0 / inverse_diag
        loo_residual = alpha / inverse_diag
        loo_nll = 0.5 * (
            math.log(2.0 * math.pi)
            + torch.log(loo_variance)
            + loo_residual.square() / loo_variance
        ).mean()
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        mll_nll = 0.5 * (
            torch.dot(y, alpha) + logdet + len(y) * math.log(2.0 * math.pi)
        ) / len(y)
        return loo_nll, mll_nll

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loo_nll, mll_nll = losses()
        loss = (1.0 - mll_weight) * loo_nll + mll_weight * mll_nll
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_loo, _ = losses()
    return float(final_loo.item())


class GenericMaternGPBackend(ModelBackend):
    name = "GP (Generic Matern)"
    family = "gaussian_process"
    supports_shap = False

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.preprocessor = _GenericPreprocessor()
        self.X_train_: torch.Tensor | None = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.final_loo_loss_: float | None = None
        self._alpha_: torch.Tensor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GenericMaternGPBackend":
        if self.config.task != "Regression":
            raise ValueError(f"{self.name} supports Regression only.")
        y_numeric = pd.to_numeric(y, errors="coerce")
        if y_numeric.isna().any():
            raise ValueError(
                f"{self.name} received missing/non-numeric target values after target preparation."
            )
        if len(y_numeric) < 3:
            raise ValueError(f"{self.name} requires at least 3 training rows.")

        X_train = self.preprocessor.fit_transform(X)
        y_values = y_numeric.to_numpy(dtype=float)
        self.y_mean_ = float(y_values.mean())
        self.y_scale_ = float(y_values.std())
        if not np.isfinite(self.y_scale_) or self.y_scale_ < 1e-12:
            self.y_scale_ = 1.0
        y_scaled = torch.as_tensor(
            (y_values - self.y_mean_) / self.y_scale_, dtype=torch.float64
        )

        torch.manual_seed(self.config.random_state)
        self.model = _ExactTabularGP(X_train.shape[1])
        self.final_loo_loss_ = _fit_exact_gp(
            self.model,
            X_train,
            y_scaled,
            max_iter=self.config.gp_max_iter,
            lr=self.config.gp_lr,
            mll_weight=self.config.gp_mll_weight,
        )
        self.X_train_ = X_train
        self._alpha_ = self._alpha(X_train, y_scaled)
        return self

    def _alpha(self, X_train: torch.Tensor, y_scaled: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            covariance = self.model.covariance(X_train)
            chol = torch.linalg.cholesky(covariance)
            return torch.cholesky_solve(y_scaled[:, None], chol).squeeze(-1)

    def predict(self, X: pd.DataFrame):
        if self.model is None or self.X_train_ is None or self._alpha_ is None:
            raise RuntimeError(f"{self.name} has not been fitted.")
        X_test = self.preprocessor.transform(X)
        prediction_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(X_test), 8192):
                batch = X_test[start : start + 8192]
                prediction_batches.append(self.model.kernel(batch, self.X_train_) @ self._alpha_)
        mean_scaled = torch.cat(prediction_batches) if prediction_batches else torch.empty(0, dtype=torch.float64)
        return mean_scaled.cpu().numpy() * self.y_scale_ + self.y_mean_
