from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False.")
    return device


def warm_tabicl(device: str) -> None:
    from tabicl import TabICLClassifier, TabICLRegressor

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(320, 8)).astype(np.float32))
    y_reg = pd.Series(0.8 * X.iloc[:, 0] - 0.3 * X.iloc[:, 1] + rng.normal(scale=0.1, size=320))
    y_clf = pd.Series(np.where(X.iloc[:, 0] + X.iloc[:, 1] > 0, "A", "B"))

    print("Caching TabICLv2 regression checkpoint...")
    TabICLRegressor(device=device, n_estimators=1, batch_size=1).fit(X, y_reg)
    print("Caching TabICLv2 classification checkpoint...")
    TabICLClassifier(device=device, n_estimators=1, batch_size=1).fit(X, y_clf)


def warm_tabfm(device: str) -> None:
    from tabfm import TabFMClassifier, TabFMRegressor
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(100, 8)).astype(np.float32))
    y_reg = pd.Series(0.8 * X.iloc[:, 0] - 0.3 * X.iloc[:, 1] + rng.normal(scale=0.1, size=100))
    y_clf = pd.Series(np.where(X.iloc[:, 0] + X.iloc[:, 1] > 0, "A", "B"))
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None

    print("Caching TabFM regression checkpoint...")
    reg_base = tabfm_v1_0_0.load(model_type="regression", device=device, dtype=dtype)
    TabFMRegressor(model=reg_base).fit(X, y_reg)

    print("Caching TabFM classification checkpoint...")
    clf_base = tabfm_v1_0_0.load(model_type="classification", device=device, dtype=dtype)
    TabFMClassifier(model=clf_base).fit(X, y_clf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download Tabtester foundation-model checkpoints.")
    parser.add_argument("--model", choices=["tabicl", "tabfm", "all"], default="tabicl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--accept-tabfm-license",
        action="store_true",
        help="Required before downloading the default TabFM pretrained weights.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device}")

    if args.model in ("tabfm", "all") and not args.accept_tabfm_license:
        raise SystemExit(
            "TabFM default pretrained weights are non-commercial/non-production. "
            "Re-run with --accept-tabfm-license after reviewing the upstream weight license."
        )

    if args.model in ("tabicl", "all"):
        warm_tabicl(device)
    if args.model in ("tabfm", "all"):
        warm_tabfm(device)
    print("Checkpoint warmup complete.")


if __name__ == "__main__":
    main()
