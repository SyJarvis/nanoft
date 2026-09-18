"""Training preparation utilities for LoRA/QLoRA fine-tuning."""

from __future__ import annotations

from typing import Union

import torch.nn as nn

from .apply import apply_lora
from .config import LoRAConfig, QLoRAConfig
from .quant import quantize_model


def prepare_model_for_training(
    model: nn.Module,
    config: Union[LoRAConfig, QLoRAConfig],
) -> nn.Module:
    """Prepare a model for LoRA or QLoRA fine-tuning.

    Steps:
        1. QLoRAConfig only: quantize base weights to 4-bit on CUDA
        2. Freeze all existing parameters
        3. Apply LoRA layers to target modules (quantized targets are
           wrapped in-place, LoRA params created in the compute dtype)
        4. Verify only LoRA params require gradients
    """
    compute_dtype = None
    if isinstance(config, QLoRAConfig):
        from .quant import _DTYPE_MAP

        model = quantize_model(model, config.quantization)
        compute_dtype = _DTYPE_MAP[config.quantization.compute_dtype]
        lora_config = config.lora
    else:
        lora_config = config

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA adapters (also handles freezing + LoRA grad enable)
    model = apply_lora(model, lora_config, compute_dtype=compute_dtype)

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    """Print count and percentage of trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
