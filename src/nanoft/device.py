"""Device detection and management for edge deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeviceInfo:
    """Information about the available compute device."""
    device_type: str       # "cuda", "mps", "cpu"
    device_name: str       # e.g. "NVIDIA Jetson AGX Orin", "Apple M2"
    total_memory_gb: float
    supports_fp16: bool
    supports_bf16: bool


def detect_device() -> DeviceInfo:
    """Detect the best available compute device.

    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        cap = torch.cuda.get_device_capability(0)
        return DeviceInfo(
            device_type="cuda",
            device_name=name,
            total_memory_gb=round(mem, 1),
            supports_fp16=True,
            supports_bf16=cap >= (8, 0),
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device_type="mps",
            device_name="Apple Metal",
            total_memory_gb=0.0,  # unified memory, not queryable from PyTorch
            supports_fp16=True,
            supports_bf16=False,  # MPS bf16 support is incomplete
        )

    import platform
    return DeviceInfo(
        device_type="cpu",
        device_name=platform.processor() or "CPU",
        total_memory_gb=0.0,
        supports_fp16=False,
        supports_bf16=False,
    )


def get_recommended_dtype(device_type: str) -> torch.dtype:
    """Return recommended compute dtype for the device."""
    if device_type == "cuda":
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def move_to_device(
    model: nn.Module,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Move model to target device with proper dtype handling."""
    if device is None:
        info = detect_device()
        device = info.device_type
    if dtype is None:
        dtype = get_recommended_dtype(device)
    return model.to(device=device, dtype=dtype)
