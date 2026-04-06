"""Merge LoRA adapter weights into base model weights."""

from __future__ import annotations

import torch.nn as nn

from .layers import LoRALinear


def merge_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """Merge all LoRA adapters into base weights.

    For LoRALinear: W_merged = W_base + (B @ A) * scaling, then delete LoRA params.
    The layer remains a LoRALinear but with merged=True, so forward() skips the
    LoRA branch and uses the standard F.linear path.

    Args:
        model: Model with LoRA layers.
        inplace: If True, modify model in-place.

    Returns:
        Model with merged weights.
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    for module in model.modules():
        if isinstance(module, LoRALinear) and module.r > 0 and not module.merged:
            module.merge()

    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """Reverse a merge operation (only possible if LoRA params still exist)."""
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.r > 0 and module.merged:
            module.unmerge()
    return model
