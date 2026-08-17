ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

ARG PREFETCH_FOUNDATION_MODELS=tabicl
ARG ACCEPT_TABFM_LICENSE=0
ARG ENABLE_JAPANESE_SUPPORT=0

LABEL org.opencontainers.image.title="Tabtester" \
      org.opencontainers.image.description="A Streamlit workbench for tabular model evaluation" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/tabtester/huggingface \
    HF_HUB_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /workspace/app

# Japanese-capable fonts are optional to avoid region-specific image bloat by default.
RUN if [ "${ENABLE_JAPANESE_SUPPORT}" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends fontconfig fonts-noto-cjk \
        && rm -rf /var/lib/apt/lists/* \
        && fc-cache -f; \
    else \
        echo "Skipping Japanese font installation."; \
    fi

COPY requirements-ngc.txt ./requirements-ngc.txt
RUN python -m pip install --no-cache-dir -r requirements-ngc.txt \
    && python - <<'PY'
import importlib.metadata
import torch

for package in ("torch", "tabicl", "tabfm", "streamlit"):
    print(f"{package}: {importlib.metadata.version(package)}")
print(f"cuda_runtime: {torch.version.cuda}")
PY

COPY scripts ./scripts
RUN mkdir -p "${HF_HOME}" \
    && case "${PREFETCH_FOUNDATION_MODELS}" in \
        none) \
            echo "Skipping foundation-model prefetch." \
            ;; \
        tabicl) \
            python scripts/warmup_models.py --model tabicl --device cpu \
            ;; \
        tabfm) \
            test "${ACCEPT_TABFM_LICENSE}" = "1" \
            && python scripts/warmup_models.py --model tabfm --device cpu --accept-tabfm-license \
            ;; \
        all) \
            test "${ACCEPT_TABFM_LICENSE}" = "1" \
            && python scripts/warmup_models.py --model all --device cpu --accept-tabfm-license \
            ;; \
        *) \
            echo "Unknown PREFETCH_FOUNDATION_MODELS=${PREFETCH_FOUNDATION_MODELS}" >&2 \
            exit 2 \
            ;; \
    esac

COPY app.py ./app.py
COPY tabtester ./tabtester
COPY LICENSE ./LICENSE

# Runtime is offline by default. All requested foundation-model weights must be in the image cache.
ENV HF_HUB_OFFLINE=1 \
    TABTESTER_PREFETCHED_MODELS=${PREFETCH_FOUNDATION_MODELS}

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["streamlit", "run", "app.py"]
