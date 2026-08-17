# Tabtester

Tabtester is a Streamlit workbench for comparing tabular foundation models with classical machine-learning baselines on the same CSV dataset.

The project currently supports regression and classification, holdout benchmarking, prediction of new rows, and missing-target imputation. The backend interface is intentionally small so additional tabular models can be added later without expanding the Streamlit app into model-specific branches.

## Current backends

- TabICLv2
- TabFM using the PyTorch backend
- XGBoost
- LightGBM
- CatBoost
- Tuned XGBoost with Optuna
- FLAML
- AutoGluon Tabular when installed separately

## Important TabFM license notice

Tabtester itself is MIT licensed. That does not change the licenses of third-party packages or pretrained weights.

The TabFM source code is Apache-2.0, but the default pretrained TabFM weights are distributed under a separate non-commercial, non-production license. Tabtester therefore:

- does not include TabFM weights in the source archive; an opt-in Docker build may embed them in the resulting image;
- shows a license warning before TabFM execution;
- requires an explicit acknowledgement in the UI before the default TabFM weights can be used;
- requires `--accept-tabfm-license` before the warmup script downloads TabFM weights.

Review the upstream TabFM weight license before use. A separately licensed local TabFM checkpoint can be supplied with `TABFM_CHECKPOINT_PATH`.

## Docker with NVIDIA NGC

The default image is:

```text
nvcr.io/nvidia/pytorch:26.05-py3
```

This image provides the PyTorch/CUDA stack. Tabtester does not reinstall PyTorch in the NGC image.

Japanese-specific compatibility is disabled by default. The default container does not install Noto CJK fonts and CSV input is limited to UTF-8/UTF-8 BOM.

To enable Japanese support, set this in `.env` before building the image:

```text
ENABLE_JAPANESE_SUPPORT=1
```

When enabled, the Docker build installs Noto CJK fonts, Matplotlib prefers a Japanese-capable font, and CSV input also accepts CP932. Rebuild the image after changing this option because the font package is selected at build time.

Requirements on the host:

- Docker Engine with Docker Compose
- NVIDIA driver compatible with the CUDA version in the selected NGC image
- NVIDIA Container Toolkit
- NGC access for pulling `nvcr.io/nvidia/pytorch`

Create the local environment file and start the app:

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

Open:

```text
http://localhost:8501
```

The Compose GPU reservation is:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Check the container GPU environment:

```bash
docker compose exec tabtester python scripts/check_gpu.py
```

## Foundation-model checkpoints and offline runtime

The benchmark timer starts immediately before `backend.fit(...)`. For the foundation-model backends, checkpoint acquisition or loading can happen inside `fit`, so an uncached first run can include network download time in `Fit Time (s)`. The recommended Docker workflow avoids that network component by downloading the requested checkpoints during image build and running the container with Hugging Face Hub offline mode enabled.

The image stores its Hugging Face cache at:

```text
/opt/tabtester/huggingface
```

This cache is part of the image rather than a host-mounted volume, so moving the built image to another compatible Docker host keeps the checkpoint files with it.

By default the build prefetches TabICLv2 only, keeps Japanese-specific compatibility disabled, and runs offline:

```text
PREFETCH_FOUNDATION_MODELS=tabicl
ACCEPT_TABFM_LICENSE=0
ENABLE_JAPANESE_SUPPORT=0
HF_HUB_OFFLINE=1
```

To bake both TabICLv2 and the default TabFM regression/classification weights into the image, first review and accept the TabFM pretrained-weight license, then set:

```text
PREFETCH_FOUNDATION_MODELS=all
ACCEPT_TABFM_LICENSE=1
HF_HUB_OFFLINE=1
```

and build normally:

```bash
docker compose build
docker compose up -d
```

The build machine needs network access to the model sources. The running container does not: `HF_HUB_OFFLINE=1` is the default runtime setting. Classical backends such as XGBoost, LightGBM, CatBoost, FLAML, and tuned XGBoost do not use pretrained foundation-model checkpoints.

`Fit Time (s)` still includes local checkpoint deserialization and model preparation performed by the backend. It excludes network transfer only when the required weights were prefetched successfully and runtime offline mode is enabled.

## Local Python installation

Docker is the recommended GPU path. A local environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

GPU-enabled local PyTorch installation may require a platform-specific PyTorch package/index. The Docker image avoids most CUDA/PyTorch version matching work. For local Python execution, set `ENABLE_JAPANESE_SUPPORT=1` in the process environment if CP932 input and Japanese plot-font selection are needed; `.env` is consumed automatically by Docker Compose, not by the standalone Python command.

## Using the app

1. Upload a CSV file.
2. Review columns excluded from target selection because they contain missing values, then choose a complete target column and optional excluded columns such as IDs.
3. Select regression or classification.
4. Select models with the individual sidebar toggles; unavailable backends remain visible but disabled.
5. Run the holdout benchmark.
6. Compare accuracy/error metrics, execution time, and prediction plots.
7. Use `Predict New Rows` to train one backend on all labeled rows and predict another CSV.
8. Use `Impute Missing Target` to predict missing values in the selected target with a foundation-model backend. The selected target is filled in place, while its pre-imputation values are preserved in an adjacent `<target>__original` backup column (or a numbered variant if that name already exists).

For regression the report includes R2, RMSE, and MAE. For classification it includes Accuracy, Balanced Accuracy, and Log Loss when probabilities and class labels are available.

## Foundation-model notes

TabICLv2 exposes device, ensemble count, batch size, AMP, KV-cache, and offload settings through Tabtester. Very small datasets are still worth testing, but TabICLv2 was pretrained on datasets starting around 300 rows, so results below that size should be validated empirically.

TabFM uses a bounded in-context window. Its estimators default to a limited number of context rows and features, so larger tables are sampled internally rather than consumed as one unlimited context. TabFM v1.0.0 classification supports at most 10 classes.

## Repository structure

```text
Tabtester/
├── app.py
├── tabtester/
│   ├── plotting.py
│   ├── utils.py
│   └── backends/
│       ├── base.py
│       ├── foundation.py
│       ├── registry.py
│       └── traditional.py
├── scripts/
│   ├── check_gpu.py
│   └── warmup_models.py
├── tests/
├── Dockerfile
├── compose.yaml
├── requirements-ngc.txt
├── requirements.txt
├── requirements-autogluon.txt
├── THIRD_PARTY.md
└── LICENSE
```

## Adding another backend

A backend implements the `ModelBackend` interface in `tabtester/backends/base.py` and is registered in `tabtester/backends/registry.py`. The Streamlit benchmark, prediction, and metric code then uses the common interface.

## Tests

The repository includes lightweight tests that do not download foundation-model checkpoints:

```bash
python -m unittest discover -s tests -v
```

## License

Tabtester source code is available under the MIT License.

Copyright (c) 2026 Tabtester contributors.

See `THIRD_PARTY.md` for third-party licensing notes.
