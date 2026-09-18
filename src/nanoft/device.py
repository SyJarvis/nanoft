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
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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


def get_recommended_dtype(
    device_type: str,
    supports_bf16: Optional[bool] = None,
) -> torch.dtype:
    """Return the stable default training dtype for the device.

    CUDA prefers BF16 when available and otherwise uses FP16. MPS defaults to
    FP32 because FP16 optimizer/training stability varies by model; callers
    with a validated low-memory MPS setup can still request FP16 explicitly.
    """
    if device_type == "cuda":
        if supports_bf16 is None:
            supports_bf16 = (
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
        return torch.bfloat16 if supports_bf16 else torch.float16
    if device_type == "mps":
        return torch.float32
    return torch.float32


def move_to_device(
    model: nn.Module,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Move model to target device with proper dtype handling."""
    supports_bf16 = None
    if device is None:
        info = detect_device()
        device = info.device_type
        supports_bf16 = info.supports_bf16
    if dtype is None:
        dtype = get_recommended_dtype(device, supports_bf16=supports_bf16)
    return model.to(device=device, dtype=dtype)
