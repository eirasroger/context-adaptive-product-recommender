"""The hardware and software a run used, recorded with the run."""

from __future__ import annotations

import os
import platform


def describe(device: str) -> dict:
    import torch

    info = {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        info["cuda"] = torch.version.cuda
    return info


def summary(info: dict) -> str:
    processor = f"{info['gpu']} ({info['gpu_memory_gb']} GB, CUDA {info['cuda']})" if "gpu" in info else "CPU only"
    return f"{processor}; {info['cpu_cores']} CPU threads; torch {info['torch']}, Python {info['python']}; {info['os']}"
