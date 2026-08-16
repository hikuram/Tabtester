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

- does not bundle TabFM weights;
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

## Model checkpoint cache

Hugging Face data is stored in a named Docker volume:

```text
tabtester_hf_cache
```

Warm only TabICLv2 checkpoints:

```bash
docker compose run --rm tabtester python scripts/warmup_models.py --model tabicl
```

Warm both TabICLv2 and the default TabFM checkpoints after reviewing the TabFM weight license:

```bash
docker compose run --rm tabtester \
  python scripts/warmup_models.py --model all --accept-tabfm-license
```

After all required checkpoints are cached, offline Hub access can be requested in `.env`:

```text
HF_HUB_OFFLINE=1
```

## Local Python installation

Docker is the recommended GPU path. A local environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

GPU-enabled local PyTorch installation may require a platform-specific PyTorch package/index. The Docker image avoids most CUDA/PyTorch version matching work.

## Using the app

1. Upload a CSV file.
2. Choose the target column and optional excluded columns such as IDs.
3. Select regression or classification.
4. Select models in the sidebar.
5. Run the holdout benchmark.
6. Compare accuracy/error metrics, execution time, and prediction plots.
7. Use `Predict New Rows` to train one backend on all labeled rows and predict another CSV.
8. Use `Impute Missing Target` to predict missing values in the selected target with a foundation-model backend.

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
