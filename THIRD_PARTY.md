# Third-party software and model notes

Tabtester's MIT License applies only to Tabtester source code and documentation authored for this repository.

## TabICL / TabICLv2

TabICL is an independent upstream project. Review its repository, model files, and license before redistribution or deployment.

Upstream repository: https://github.com/soda-inria/tabicl

## TabFM

The source repository does not ship TabFM pretrained weights. If `PREFETCH_FOUNDATION_MODELS` is set to `tabfm` or `all` and `ACCEPT_TABFM_LICENSE=1` during a Docker build, the resulting local image will contain cached TabFM weights. Redistribution or production use of such an image must comply with the separate TabFM pretrained-weight license.

The TabFM source repository uses Apache-2.0. The default pretrained weights are distributed under a separate `tabfm-non-commercial-v1.0` license and are restricted to non-commercial, non-production use.

Upstream repository: https://github.com/google-research/tabfm

The Tabtester source archive does not redistribute TabFM pretrained weights. An opt-in locally built image may contain cached weights as described above.

## NVIDIA NGC PyTorch image

The NVIDIA NGC PyTorch container is distributed by NVIDIA and is subject to NVIDIA's applicable licenses and terms. It is not covered by Tabtester's MIT License.

## Other Python dependencies

XGBoost, LightGBM, CatBoost, Optuna, FLAML, SHAP, Streamlit, PyTorch, pandas, NumPy, scikit-learn, Matplotlib, AutoGluon, and their transitive dependencies remain subject to their own licenses.


## Noto CJK fonts

When `ENABLE_JAPANESE_SUPPORT=1` is selected at Docker build time, the Dockerfile installs the distribution-provided Noto CJK font package so Matplotlib can render Japanese labels without depending on host fonts. The package is not installed by the default build. The fonts remain subject to their upstream license and distribution package notices.
