from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
import torch


MODEL_SET = os.getenv("PREFETCH_FOUNDATION_MODELS", "tabicl").strip().lower()
ACCEPT_TABFM_LICENSE = os.getenv("ACCEPT_TABFM_LICENSE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEVICE = "cpu"


def warm_tabicl(device: str) -> None:
    from tabicl import TabICLClassifier, TabICLRegressor

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(320, 8)).astype(np.float32))
    y_reg = pd.Series(0.8 * X.iloc[:, 0] - 0.3 * X.iloc[:, 1] + rng.normal(scale=0.1, size=320))
    y_clf = pd.Series(np.where(X.iloc[:, 0] + X.iloc[:, 1] > 0, "A", "B"))

    print("Caching TabICLv2 regression checkpoint...")
    reg = TabICLRegressor(device=device, n_estimators=1, batch_size=1)
    reg.fit(X, y_reg)
    del reg
    gc.collect()

    print("Caching TabICLv2 classification checkpoint...")
    clf = TabICLClassifier(device=device, n_estimators=1, batch_size=1)
    clf.fit(X, y_clf)
    del clf
    gc.collect()


def warm_tabfm(device: str) -> None:
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    print("Caching TabFM regression checkpoint...")
    reg_base = tabfm_v1_0_0.load(model_type="regression", device=device, dtype=None)
    del reg_base
    gc.collect()

    print("Caching TabFM classification checkpoint...")
    clf_base = tabfm_v1_0_0.load(model_type="classification", device=device, dtype=None)
    del clf_base
    gc.collect()


def main() -> None:
    if MODEL_SET not in {"none", "tabicl", "tabfm", "all"}:
        raise SystemExit(
            "PREFETCH_FOUNDATION_MODELS in compose.yaml must be one of: "
            "none, tabicl, tabfm, all."
        )

    print(f"device={DEVICE}")
    print(f"models={MODEL_SET}")

    if MODEL_SET in {"tabfm", "all"} and not ACCEPT_TABFM_LICENSE:
        raise SystemExit(
            "TabFM default pretrained weights are non-commercial/non-production. "
            "Set ACCEPT_TABFM_LICENSE to 1 in compose.yaml only after reviewing "
            "the upstream weight license."
        )

    if MODEL_SET == "none":
        print("No foundation models requested.")
        return
    if MODEL_SET in {"tabicl", "all"}:
        warm_tabicl(DEVICE)
    if MODEL_SET in {"tabfm", "all"}:
        warm_tabfm(DEVICE)
    print("Model prefetch complete.")


if __name__ == "__main__":
    main()
