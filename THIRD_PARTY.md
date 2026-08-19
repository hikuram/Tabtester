# Third-party software and model notes

Tabtester's MIT License applies only to Tabtester source code and documentation authored for this repository.

## TabICL / TabICLv2

TabICL is an independent upstream project. Review its repository, model files, and license before redistribution or deployment.

Upstream repository: https://github.com/soda-inria/tabicl

## TabFM

The source repository and Docker image do not ship TabFM pretrained weights. The default `compose.yaml` prefetch configuration does not request TabFM weights. To enable TabFM prefetch, the operator must explicitly edit the fixed `PREFETCH_FOUNDATION_MODELS` and `ACCEPT_TABFM_LICENSE` values in `compose.yaml` after reviewing the upstream weight license.

The TabFM source repository uses Apache-2.0. The default pretrained weights are distributed under a separate `tabfm-non-commercial-v1.0` license and are restricted to non-commercial, non-production use.

Upstream repository: https://github.com/google-research/tabfm

The Tabtester source archive and application image do not redistribute TabFM pretrained weights. The optional local model cache may contain weights downloaded by the user.

## NVIDIA NGC PyTorch image

The NVIDIA NGC PyTorch container is distributed by NVIDIA and is subject to NVIDIA's applicable licenses and terms. It is not covered by Tabtester's MIT License.

## Other Python dependencies

XGBoost, LightGBM, CatBoost, Optuna, FLAML, SHAP, Streamlit, PyTorch, pandas, NumPy, scikit-learn, Matplotlib, AutoGluon, and their transitive dependencies remain subject to their own licenses.

## Noto CJK fonts

When the Dockerfile build capability `ENABLE_JAPANESE_SUPPORT` is set to `1`, the image installs the distribution-provided Noto CJK font package so Matplotlib can render Japanese labels without depending on host fonts. The default build leaves this package out. The fonts remain subject to their upstream license and distribution package notices.
