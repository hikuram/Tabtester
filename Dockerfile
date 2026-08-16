ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Tabtester" \
      org.opencontainers.image.description="A Streamlit workbench for tabular model evaluation" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /workspace/app

COPY requirements-ngc.txt ./requirements-ngc.txt
RUN python -m pip install --no-cache-dir -r requirements-ngc.txt \
    && python - <<'PY'
import importlib.metadata
import torch

for package in ("torch", "tabicl", "tabfm", "streamlit"):
    print(f"{package}: {importlib.metadata.version(package)}")
print(f"cuda_runtime: {torch.version.cuda}")
PY

COPY app.py ./app.py
COPY tabtester ./tabtester
COPY scripts ./scripts
COPY LICENSE ./LICENSE

RUN mkdir -p /models/huggingface

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["streamlit", "run", "app.py"]
