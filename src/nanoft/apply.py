"""Apply and remove LoRA adapters from a model."""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from .config import LoRAConfig
from .layers import LoRALinear


def get_target_module_names(
    model: nn.Module,
    patterns: List[str],
) -> List[str]:
    """Return fully-qualified names of modules matching any substring pattern."""
    matched = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p in name for p in patterns):
            matched.append(name)
    return matched


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Replace target linear layers with LoRA-wrapped versions.

    Args:
        model: Any nn.Module (typically a HuggingFace model).
        config: LoRAConfig specifying rank, alpha, dropout.
        target_modules: Substring patterns to match layer names.
            Defaults to config.get_target_modules().

    Returns:
        The model with LoRA layers injected (modified in-place).
    """
    patterns = target_modules or config.get_target_module_names() if hasattr(config, 'get_target_module_names') else config.get_target_modules()
    target_names = get_target_module_names(model, patterns)

    for name in target_names:
        module = model.get_submodule(name)
        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            fan_in_fan_out=config.fan_in_fan_out,
            merge_weights=False,
        )

        # Move to same device and dtype as original module
        lora_layer = lora_layer.to(
            device=module.weight.device, dtype=module.weight.dtype
        )

        # Copy original weights
        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.bias.data.copy_(module.bias.data)

        # Sync LoRA parameter dtypes to match base weight
        lora_layer.lora_A.data = lora_layer.lora_A.data.to(module.weight.dtype)
        lora_layer.lora_B.data = lora_layer.lora_B.data.to(module.weight.dtype)

        # Enable gradient for LoRA parameters
        lora_layer.lora_A.requires_grad = True
        lora_layer.lora_B.requires_grad = True

        # Replace in model
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, lora_layer)

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

    return model


def remove_lora(model: nn.Module) -> nn.Module:
    """Remove all LoRA layers, restoring original nn.Linear layers.

    The restored Linear will hold the merged weight if LoRA was merged,
    or the original frozen weight if unmerged.
    """
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            modules_to_replace.append((name, module))

    for name, lora_module in modules_to_replace:
        # Ensure merged before extracting
        if not lora_module.merged and lora_module.r > 0:
            lora_module.merge()

        original = nn.Linear(
            in_features=lora_module.in_features,
            out_features=lora_module.out_features,
            bias=lora_module.bias is not None,
        )
        original.weight.data = lora_module._T(lora_module.weight.data).clone()
        if lora_module.bias is not None:
            original.bias.data = lora_module.bias.data.clone()

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, original)

    return model
