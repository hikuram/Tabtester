# Tabtester

Tabtester is a Streamlit workbench for comparing tabular foundation models with classical machine-learning baselines on the same CSV dataset.

The project supports regression and classification, holdout benchmarking, prediction of new rows, missing-target imputation, sequential benchmarking of multiple target columns, and data-grounded candidate recommendation for numeric target properties. The backend interface is intentionally small so additional tabular models can be added without expanding the Streamlit app into model-specific branches.

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

- does not include TabFM weights in the source archive or application image;
- stores downloaded checkpoints separately in the persistent model cache;
- shows a license warning before TabFM execution;
- requires an explicit acknowledgement in the UI before the default TabFM weights can be used;
- keeps TabFM prefetch disabled in the default Compose configuration.

Review the upstream TabFM weight license before use. A separately licensed local TabFM checkpoint can be supplied from the application UI.

## Docker with NVIDIA NGC

The image is based on:

```text
nvcr.io/nvidia/pytorch:26.05-py3
```

The base image, application port, Japanese support, cache paths, offline runtime mode, and model-prefetch defaults are intentionally written directly in `Dockerfile` or `compose.yaml`. This repository does not use a `.env` file or Compose `${...}` interpolation.

Japanese support is a build-time image choice controlled by one value in `Dockerfile`:

```dockerfile
ARG ENABLE_JAPANESE_SUPPORT=0
```

The default general-purpose build keeps it disabled. To build a Japanese-capable image, edit that line to `1` before building. When enabled, the Dockerfile installs Noto CJK and writes the Matplotlib font configuration into the image; when disabled, neither is added. The Python application contains no Japanese-support flag, font probing, generated capability file, or configuration module.

CSV decoding is independent of the image font choice: CP932 is a standard Python codec and remains available in both builds. The build option only controls whether the container includes Japanese-capable plotting fonts and Matplotlib defaults.

Requirements on the host:

- Docker Engine with Docker Compose
- NVIDIA driver compatible with the CUDA version in the selected NGC image
- NVIDIA Container Toolkit
- NGC access for pulling `nvcr.io/nvidia/pytorch`

Build the image, prefetch the default TabICLv2 checkpoints once, and start the app:

```bash
docker compose build
docker compose run --rm model-prefetch
docker compose up -d
```

The prefetch step writes model files into the persistent `model-cache` named volume. Rebuilding the image does not delete this volume.

Open:

```text
http://localhost:8501
```

Check the container GPU environment:

```bash
docker compose exec tabtester python scripts/check_gpu.py
```

## Fixed deployment configuration

The normal deployment profile is deliberately simple and reproducible.

`Dockerfile` fixes:

- NGC base image
- Japanese support build capability (`ENABLE_JAPANESE_SUPPORT=0` or `1`)
- Hugging Face, Torch, and XDG cache locations
- offline Hugging Face runtime
- Streamlit listen address and port
- Matplotlib cache directory

`compose.yaml` fixes:

- host port `8501`
- persistent `/models` cache volume
- GPU reservation
- offline network policy for the app service
- network-enabled model-prefetch service
- default prefetch target `tabicl`
- default TabFM prefetch license acknowledgement `0`

If a different deployment profile is needed, edit `Dockerfile` or `compose.yaml` explicitly and rebuild/recreate the service. There is no secondary `.env` layer that can silently change the image or service configuration.

## Foundation-model checkpoints and offline runtime

Foundation-model downloads are separated from the Docker image build. The image contains the application and Python dependencies, while model files live in the Compose named volume `model-cache`, mounted at `/models`.

The application runs with Hugging Face offline mode enabled. The `model-prefetch` service temporarily enables network access, downloads the requested checkpoints into the shared volume, and exits.

The default `compose.yaml` contains:

```yaml
PREFETCH_FOUNDATION_MODELS: "tabicl"
ACCEPT_TABFM_LICENSE: "0"
```

To prefetch TabFM as well, review its pretrained-weight license and then edit those two fixed values in `compose.yaml` to:

```yaml
PREFETCH_FOUNDATION_MODELS: "all"
ACCEPT_TABFM_LICENSE: "1"
```

Then run:

```bash
docker compose run --rm model-prefetch
```

After the cache is populated, return `compose.yaml` to the desired fixed deployment values if necessary and start the application normally.

If a required checkpoint is missing, the offline application service fails that foundation backend instead of silently downloading at benchmark time.

`docker compose down` keeps the named model cache. `docker compose down -v` removes it.

## CSV input

CSV input uses compact Encoding and Delimiter selectors. Auto mode handles UTF-8, UTF-8 BOM, BOM-marked UTF-16, and CP932 without extra charset-detection dependencies, together with comma, tab, and semicolon separators. CP932 decoding does not depend on the Japanese plotting build option.

## Using the app

1. Upload a CSV file. Encoding and delimiter default to Auto; open `CSV input options` only when manual selection is needed.
2. Choose one or more complete target columns and optional excluded columns such as IDs. All selected target columns are removed from the feature set for every benchmark target to reduce target leakage.
3. Select regression or classification.
4. Select models with the individual sidebar toggles; unavailable backends remain visible but disabled.
5. Run the holdout benchmark. Selected targets are processed sequentially; a failed model or target does not stop the remaining targets.
6. Review the benchmark overview and download the primary `Download complete results (.zip)` artifact when a run should be retained. Individual target CSV downloads remain available.
7. Open target results in completion-order pages for detailed inspection.
8. Use `Predict New Rows` to train one backend on all labeled rows and predict another CSV with the same leakage-safe feature set.
9. Use `Impute Missing Target` to fill missing target values in place. The pre-imputation target is preserved in an adjacent `<target>__original` backup column, with a numbered suffix if required.
10. Use `Recommend Candidates` for numeric-property inverse design. Define target goals, editable search ranges, optional discrete steps, and an optional fixed-sum mixture constraint. Tabtester screens candidate conditions with multiple regression backends, reports consensus and model disagreement, and returns a Pareto table plus a diverse shortlist. Recommendation also records total execution time, phase-level timing, and per-model/per-property fit and prediction time so expensive searches can be diagnosed.

For regression the report includes R2, RMSE, and MAE. For classification it includes Accuracy, Balanced Accuracy, and Log Loss when probabilities and class labels are available.


## Complete benchmark export

Each completed Benchmark and Evaluation run can be downloaded as one ZIP archive without rerunning any model. The complete ZIP is the primary benchmark download, while the existing per-target prediction CSV downloads remain available for quick access.

The archive contains:

```text
tabtester_benchmark_YYYYMMDD_HHMMSS_UTC.zip
├── run_settings.csv
├── benchmark_summary.csv
├── failures.csv
├── 01_<target>/
│   ├── metrics.csv
│   ├── predictions.csv
│   ├── metric_comparison.png
│   ├── execution_time.png
│   ├── actual_vs_predicted.png      # regression
│   ├── confusion_matrix.png         # classification
│   ├── shap_<model>.png             # when available
│   └── errors.csv                   # when needed
├── export_errors.csv                # only if an artifact could not be rendered
└── ...
```

`run_settings.csv` is stored as `Section, Setting, Value` rows so new settings can be added without changing a wide one-row schema. It records the run timestamps and elapsed time, application source fingerprint, source CSV filename and SHA256, parser settings, dataset shape and column names, missing-cell count, task, all target and excluded columns, selected models, train/test fractions, the random seed, leakage-guard policy, foundation-model settings, Optuna/AutoML settings, per-target split sizes, and relevant Python/package/CUDA/GPU information.

`benchmark_summary.csv` contains one row per target/model evaluation, including metrics, fit/predict time, status, and error text. `failures.csv` keeps failed target/model evaluations visible rather than silently dropping them. The uploaded source CSV itself is not copied into the archive.

## Candidate recommendation

Recommendation is intended as a data-grounded second opinion for selecting the next conditions to test, not as an autonomous claim of a globally optimal formulation.

The workflow supports:

- multiple numeric target properties with `Range`, `At least`, `At most`, `Close to`, `Maximize`, or `Minimize` goals;
- low/medium/high priority and optional hard constraints;
- editable search minimum, search maximum, and optional step for each active design variable;
- a fixed-sum mixture constraint such as `A + B + Filler = 100`;
- automatic candidate-count suggestions based on effective dimensionality and search-range width;
- exhaustive enumeration for small fully discrete spaces and space-filling sampling otherwise;
- optional automatic search expansion while the top shortlist is still improving;
- median consensus across selected models, with model disagreement reported separately;
- in-domain, near-edge, and extrapolation diagnostics;
- Pareto candidates, a diversity-filtered shortlist, and nearest existing experiments for candidate review.

TabFM is intentionally excluded from Recommendation when using the default pretrained weights because those weights are restricted to non-commercial, non-production use. TabFM remains available in Benchmark after explicit license acknowledgement.

Selected Benchmark target columns remain protected from use as Recommendation model inputs, so adding Recommendation does not weaken the existing target-leakage guard.

## Local Python installation

Docker is the recommended GPU path. A local environment can still be created with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

A direct local run uses the same CSV parser, including CP932. Plot fonts are left to the local Matplotlib/OS configuration; Tabtester does not probe or alter them. The supported deployment profile is the Docker image described above.

## Repository structure

```text
Tabtester/
├── app.py
├── tabtester/
│   ├── plotting.py
│   ├── export.py
│   ├── utils.py
│   ├── recommendation.py
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

A backend implements the `ModelBackend` interface in `tabtester/backends/base.py` and is registered in `tabtester/backends/registry.py`. The Streamlit benchmark, prediction, recommendation, and metric code then uses the common interface.

## Tests

The repository includes lightweight tests that do not download foundation-model checkpoints:

```bash
python -m unittest discover -s tests -v
```

## License

Tabtester source code is available under the MIT License.

Copyright (c) 2026 Tabtester contributors.

See `THIRD_PARTY.md` for third-party licensing notes.
