"""Device selection and diagnostic information."""
import torch


def device_info() -> dict:
    available = torch.cuda.is_available()
    info = {"torch_version": str(torch.__version__), "cuda_available": available,
            "device": "cuda" if available else "cpu", "gpu_model": None,
            "gpu_memory_bytes": None}
    if available:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update(gpu_model=properties.name, gpu_memory_bytes=properties.total_memory)
    return info


def detect_device() -> torch.device:
    """Print diagnostics before an experiment and return the selected device."""
    info = device_info()
    print(f"PyTorch version: {info['torch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"GPU model: {info['gpu_model']}")
        print(f"GPU memory: {info['gpu_memory_bytes'] / 2**30:.2f} GiB")
    print(f"Selected device: {info['device']}")
    return torch.device(info["device"])
