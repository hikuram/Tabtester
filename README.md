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

Japanese-specific compatibility is disabled by default. CSV input supports UTF-8, UTF-8 BOM, and BOM-marked UTF-16 without extra dependencies. The CSV parser also exposes compact Encoding and Delimiter selectors, both defaulting to Auto.

To enable Japanese support, set this in `.env` before building the image:

```text
ENABLE_JAPANESE_SUPPORT=1
```

When enabled, the Docker build installs Noto CJK fonts, Matplotlib prefers a Japanese-capable font, and CP932 becomes available for CSV input. Auto encoding detection tries CP932 only in this mode. Tabtester uses a dedicated Matplotlib cache and directly registers installed Japanese font files if the cached font list does not contain them. Rebuild the image after changing this option because the font package is selected at build time. The Environment panel should report `Matplotlib font: Noto Sans CJK JP` (or another detected Japanese-capable font).

Requirements on the host:

- Docker Engine with Docker Compose
- NVIDIA driver compatible with the CUDA version in the selected NGC image
- NVIDIA Container Toolkit
- NGC access for pulling `nvcr.io/nvidia/pytorch`

Create the local environment file, build the app image, prefetch the default foundation-model checkpoints once, and start the app:

```bash
cp .env.example .env
docker compose build
docker compose run --rm model-prefetch
docker compose up -d
```

The prefetch step writes model files into the persistent `model-cache` named volume. Rebuilding the image does not delete this volume.

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

Foundation-model downloads are intentionally separated from the Docker image build. The image contains the application and Python dependencies, while model files live in the Compose named volume `model-cache`, mounted at:

```text
/models
```

The container uses these cache locations:

```text
HF_HOME=/models/huggingface
TORCH_HOME=/models/torch
XDG_CACHE_HOME=/models/xdg
```

By default `model-prefetch` downloads TabICLv2 into that volume:

```text
PREFETCH_FOUNDATION_MODELS=tabicl
ACCEPT_TABFM_LICENSE=0
HF_HUB_OFFLINE=1
```

Run prefetch explicitly whenever a new cache is needed or the requested model set changes:

```bash
docker compose run --rm model-prefetch
```

The prefetch service uses the same `tabtester:local` image as the app, enables Hugging Face network access only for that run, and exits when checkpoint preparation is complete. Repeated image builds reuse the existing named volume and therefore do not bake the large model files into each image.

To prefetch TabICLv2 plus the default TabFM regression/classification weights, first review and accept the TabFM pretrained-weight license, then set:

```text
PREFETCH_FOUNDATION_MODELS=all
ACCEPT_TABFM_LICENSE=1
```

and run:

```bash
docker compose run --rm model-prefetch
```

Runtime remains offline by default. With the requested checkpoints already present in `model-cache`, start the application normally:

```bash
docker compose up -d
```

If a required checkpoint is missing while `HF_HUB_OFFLINE=1`, the corresponding foundation backend will fail rather than silently downloading during a benchmark. Set `HF_HUB_OFFLINE=0` only when network fallback is intentionally desired.

`docker compose down` keeps the named model cache. `docker compose down -v` removes it and should only be used when the cached model files are intentionally being discarded.

The benchmark timer starts immediately before `backend.fit(...)`. `Fit Time (s)` therefore still includes local checkpoint deserialization and model preparation, but not network transfer when the persistent cache is populated and runtime offline mode is enabled.

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

1. Upload a CSV file. Encoding and delimiter default to Auto; open `CSV input options` only when manual selection is needed. Auto handles UTF-8, UTF-8 BOM, BOM-marked UTF-16, and common comma/tab/semicolon separators. CP932 is included when Japanese support is enabled.
2. Review columns excluded from target selection because they contain missing values, then choose one or more complete target columns and optional excluded columns such as IDs. All selected target columns are removed from the feature set for every benchmark target to reduce target leakage.
3. Select regression or classification.
4. Select models with the individual sidebar toggles; unavailable backends remain visible but disabled.
5. Run the holdout benchmark. Selected targets are processed sequentially; a failed model or target does not stop the remaining targets.
6. Review the benchmark overview, then open target results in completion-order pages to compare metrics, execution time, prediction plots, and per-model errors.
7. Use `Predict New Rows` to choose one of the selected benchmark targets, train one backend on all labeled rows, and predict another CSV using the same leakage-safe feature set.
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
