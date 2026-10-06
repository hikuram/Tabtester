from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd


MODEL_SET_RAW = os.getenv("PREFETCH_FOUNDATION_MODELS", "tabicl,kumo").strip().lower()
KUMO_SIZES_RAW = os.getenv("KUMO_PREFETCH_SIZES", "small").strip().lower()
ACCEPT_TABFM_LICENSE = os.getenv("ACCEPT_TABFM_LICENSE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEVICE = "cpu"


def _parse_model_set(value: str) -> set[str]:
    if value == "none":
        return set()
    if value == "all":
        return {"tabicl", "kumo", "tabfm"}
    models = {item.strip() for item in value.split(",") if item.strip()}
    unknown = models - {"tabicl", "kumo", "tabfm"}
    if unknown:
        raise SystemExit(
            "PREFETCH_FOUNDATION_MODELS must contain only tabicl, kumo, tabfm, "
            "or use none/all. Unknown: " + ", ".join(sorted(unknown))
        )
    return models


def _parse_kumo_sizes(value: str) -> list[str]:
    if value == "all":
        return ["small", "medium", "large"]
    sizes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(sizes) - {"small", "medium", "large"}
    if not sizes or unknown:
        raise SystemExit(
            "KUMO_PREFETCH_SIZES must contain small, medium, large, or all."
        )
    return sizes


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


def warm_kumo(device: str, sizes: list[str]) -> None:
    import sdm

    for size in sizes:
        for task in ("regression", "classification"):
            print(f"Caching KumoTabular {size} {task} checkpoint...")
            model = sdm.models.KumoTabular(task=task, size=size, device=device)
            del model
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
    models = _parse_model_set(MODEL_SET_RAW)
    kumo_sizes = _parse_kumo_sizes(KUMO_SIZES_RAW)

    print(f"device={DEVICE}")
    print(f"models={','.join(sorted(models)) or 'none'}")
    if "kumo" in models:
        print(f"kumo_sizes={','.join(kumo_sizes)}")

    if "tabfm" in models and not ACCEPT_TABFM_LICENSE:
        raise SystemExit(
            "TabFM default pretrained weights are non-commercial/non-production. "
            "Set ACCEPT_TABFM_LICENSE to 1 in compose.yaml only after reviewing "
            "the upstream weight license."
        )

    if not models:
        print("No foundation models requested.")
        return
    if "tabicl" in models:
        warm_tabicl(DEVICE)
    if "kumo" in models:
        warm_kumo(DEVICE, kumo_sizes)
    if "tabfm" in models:
        warm_tabfm(DEVICE)
    print("Model prefetch complete.")


if __name__ == "__main__":
    main()
