"""Merge LoRA adapter weights into base model weights."""

from __future__ import annotations

import copy

import torch.nn as nn

from .layers import LoRALinear, is_lora_module
from .quant import LoRAQuantLinear


def _reject_quantized(model: nn.Module) -> None:
    quantized = [
        name for name, module in model.named_modules()
        if isinstance(module, LoRAQuantLinear)
    ]
    if quantized:
        raise TypeError(
            "cannot merge into 4-bit quantized weights; use "
            "nanoft.quant.dequantize_and_merge() instead: "
            + ", ".join(quantized)
        )


def merge_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """Merge all LoRA adapters into base weights.

    For LoRALinear: W_merged = W_base + (B @ A) * scaling, then delete LoRA params.
    The layer remains a LoRALinear but with merged=True, so forward() skips the
    LoRA branch and uses the standard F.linear path.

    Raises TypeError if the model contains quantized LoRA layers — 4-bit
    weights cannot be merged in place (use dequantize_and_merge()).

    Args:
        model: Model with LoRA layers.
        inplace: If True, modify model in-place.

    Returns:
        Model with merged weights.
    """
    _reject_quantized(model)

    if not inplace:
        import copy
        model = copy.deepcopy(model)

    for module in model.modules():
        if is_lora_module(module) and module.r > 0 and not module.merged:
            module.merge()

    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """Reverse a merge operation (only possible if LoRA params still exist)."""
    _reject_quantized(model)
    for module in model.modules():
        if is_lora_module(module) and module.r > 0 and module.merged:
            module.unmerge()
    return model


def merge_and_unload_lora(model: nn.Module, inplace: bool = True) -> nn.Module:
    """Merge LoRA weights and replace LoRA layers with plain ``nn.Linear``.

    This is the right path before saving a full transformers model: the returned
    model has ordinary Linear layers and no ``lora_A`` / ``lora_B`` parameters.

    Raises TypeError if the model contains quantized LoRA layers — 4-bit
    weights cannot be merged in place (use dequantize_and_merge()).

    Args:
        model: Model with LoRA layers.
        inplace: If True, modify model in-place. If False, deep-copy first.

    Returns:
        Model with LoRA deltas merged into base weights and adapters unloaded.
    """
    _reject_quantized(model)

    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace: list[tuple[str, LoRALinear]] = []
    for name, module in model.named_modules():
        if is_lora_module(module):
            modules_to_replace.append((name, module))

    for name, lora_module in modules_to_replace:
        if lora_module.r > 0 and not lora_module.merged:
            lora_module.merge()

        linear = nn.Linear(
            in_features=lora_module.in_features,
            out_features=lora_module.out_features,
            bias=lora_module.bias is not None,
            device=lora_module.weight.device,
            dtype=lora_module.weight.dtype,
        )
        linear.weight.data.copy_(lora_module._T(lora_module.weight.data))
        if lora_module.bias is not None:
            linear.bias.data.copy_(lora_module.bias.data)
        linear.weight.requires_grad = lora_module.weight.requires_grad
        if linear.bias is not None and lora_module.bias is not None:
            linear.bias.requires_grad = lora_module.bias.requires_grad

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, linear)

    return model
