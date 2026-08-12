import torch

print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    memory_gib = props.total_memory / (1024 ** 3)
    print(f"gpu[{idx}]={props.name}, vram={memory_gib:.1f} GiB")
