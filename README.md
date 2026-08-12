# Tabtester

Tabtester is a lightweight Streamlit workbench for quickly evaluating tabular machine-learning models.
The current backend is TabICLv2. The project is intentionally small and is expected to expand with additional models, evaluation methods, and comparison tools over time.

## Current features

- Upload a training CSV.
- Select the target column and exclude ID-like columns.
- Run regression or classification with TabICLv2.
- Run a simple holdout evaluation.
- Predict rows from a second CSV.
- Use CPU or CUDA inference.
- Display the detected PyTorch/CUDA environment.
- Use a local checkpoint path for offline or internal environments.
- Read UTF-8, UTF-8 with BOM, and CP932 CSV files.
- Export UTF-8 with BOM CSV files for convenient Excel use.

## Recommended deployment: NGC PyTorch + Docker Compose

The default base image is:

```text
nvcr.io/nvidia/pytorch:24.12-py3
```

The container keeps the CUDA/PyTorch stack provided by the NVIDIA NGC image and installs TabICL and Streamlit on top.
You can change the base image through `BASE_IMAGE` in `.env` without editing the Dockerfile.

### Host requirements

- Linux host with an NVIDIA GPU
- Docker Engine with Docker Compose
- NVIDIA GPU driver compatible with the selected NGC image
- NVIDIA Container Toolkit configured for Docker

A separate CUDA Toolkit installation on the host is not required for this container.

### Build and start

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

Open:

```text
http://localhost:8501
```

Stop the service with:

```bash
docker compose down
```

### GPU access

The Compose file requests all NVIDIA GPUs with:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Verify GPU access inside the running container:

```bash
docker compose exec tabtester python scripts/check_gpu.py
```

You should see `cuda_available=True` and at least one GPU name.

## Checkpoint cache and offline use

The Hugging Face cache is stored in the named Docker volume `tabtester_hf_cache`, so rebuilding the application image does not normally require downloading the checkpoints again.

Pre-download the TabICLv2 checkpoints while external network access is available:

```bash
docker compose run --rm tabtester python scripts/warmup_models.py
```

After the cache is populated, set the following in `.env` for offline operation:

```text
HF_HUB_OFFLINE=1
```

Then restart the service.

## GPU-oriented settings

For a modest GPU, a practical starting point is:

- Ensemble size: 8
- Batch size: 4
- Device: auto or cuda
- KV cache: off for one-off evaluation

If GPU memory is tight, reduce the batch size to 1 or 2 first. Reducing the ensemble size to 4 can also improve responsiveness.
KV cache can be useful when repeatedly predicting new rows against the same training table, at the cost of additional memory.

## Native Python alternative

Python 3.10 or newer is required by the current TabICL dependency.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

## Project status and roadmap


## Notes

- `requirements-ngc.txt` intentionally does not install PyTorch. The PyTorch build supplied by the NGC base image is preserved.
- The first TabICL fit downloads its pretrained checkpoint unless it is already cached or a local checkpoint path is supplied.
- Very small datasets should be validated carefully. Benchmark performance does not guarantee accuracy for a specific dataset or domain.

## License

Tabtester source code is released under the MIT License. See [LICENSE](LICENSE).

Copyright (c) 2026 Tabtester contributors.

Third-party libraries, model implementations, pretrained weights, container images, and other dependencies remain subject to their own licenses and terms. The Tabtester MIT License does not relicense those third-party components. Review the applicable upstream terms before redistribution or commercial use.
