from __future__ import annotations

import importlib.metadata

import torch


for package in ("torch", "tabicl", "tabfm", "streamlit"):
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package}={version}")

print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        vram_gib = props.total_memory / (1024 ** 3)
        print(f"gpu[{index}]={props.name}, vram={vram_gib:.1f} GiB")
