FROM nvcr.io/nvidia/pytorch:26.05-py3


ARG ENABLE_JAPANESE_SUPPORT=0

LABEL org.opencontainers.image.title="Tabtester" \
      org.opencontainers.image.description="A Streamlit workbench for tabular model evaluation" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/xdg \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TABTESTER_MODEL_CACHE=/models \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/opt/tabtester/matplotlib

WORKDIR /workspace/app

# Japanese support is a build-only image choice. Edit the default above to 1
# when building the Japanese-capable image. No runtime setting is used.
RUN set -eu; \
    case "${ENABLE_JAPANESE_SUPPORT}" in \
        0|1) ;; \
        *) echo "ENABLE_JAPANESE_SUPPORT must be 0 or 1." >&2; exit 2 ;; \
    esac; \
    rm -rf "${MPLCONFIGDIR}"; \
    mkdir -p "${MPLCONFIGDIR}"; \
    if [ "${ENABLE_JAPANESE_SUPPORT}" = "1" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends fontconfig fonts-noto-cjk; \
        rm -rf /var/lib/apt/lists/*; \
        fc-cache -f; \
        printf 'font.family: sans-serif\nfont.sans-serif: Noto Sans CJK JP, DejaVu Sans\naxes.unicode_minus: False\n' > "${MPLCONFIGDIR}/matplotlibrc"; \
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
COPY app.py ./app.py
COPY tabtester ./tabtester
COPY LICENSE ./LICENSE

# Foundation-model weights are stored in the runtime model-cache volume.
# Use the model-prefetch Compose service before offline execution.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["streamlit", "run", "app.py"]
