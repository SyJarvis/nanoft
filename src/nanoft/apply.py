"""Apply and remove LoRA adapters from a model."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import LoRAConfig
from .layers import LoRALinear, is_lora_module
from .quant import LoRAQuantLinear, is_quantized_linear


def get_target_module_names(
    model: nn.Module,
    patterns: List[str],
) -> List[str]:
    """Return Linear module names matching exact names or dotted suffixes."""
    matched = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name == pattern or name.endswith(f".{pattern}")
            for pattern in patterns
        ):
            matched.append(name)
    return matched


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: Optional[List[str]] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Replace target linear layers with LoRA-wrapped versions.

    Dense ``nn.Linear`` targets become ``LoRALinear`` (weights copied);
    quantized Linear targets (e.g. ``bnb.nn.Linear4bit``) are wrapped
    in-place by ``LoRAQuantLinear`` — their 4-bit weight is never copied
    or modified, and the LoRA parameters are created in *compute_dtype*
    (default: torch's default dtype, e.g. fp32 for CPU-side tests).

    Args:
        model: Any nn.Module (typically a HuggingFace model).
        config: LoRAConfig specifying rank, alpha, dropout.
        target_modules: Exact module names or dotted suffixes to match.
            Defaults to config.get_target_modules().
        compute_dtype: dtype for LoRA parameters on quantized targets
            (QLoRA compute dtype). Ignored for dense targets, whose LoRA
            parameters follow the base weight dtype.

    Returns:
        The model with LoRA layers injected (modified in-place).
    """
    existing_lora = [
        name for name, module in model.named_modules()
        if is_lora_module(module)
    ]
    if existing_lora:
        raise ValueError(
            "Model already contains LoRA layers; repeated injection is not supported: "
            + ", ".join(existing_lora)
        )

    patterns = (
        config.get_target_modules()
        if target_modules is None
        else target_modules
    )
    if not patterns:
        raise ValueError("target_modules must be a non-empty list")

    target_names = get_target_module_names(model, patterns)
    if not target_names:
        raise ValueError(
            "No nn.Linear modules matched target_modules: "
            + ", ".join(patterns)
        )

    for name in target_names:
        module = model.get_submodule(name)

        if is_quantized_linear(module):
            # QLoRA path: wrap the quantized layer, never touch its weight.
            lora_layer = LoRAQuantLinear(
                base_layer=module,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                compute_dtype=compute_dtype,
            )
            lora_layer.train(module.training)

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
            continue

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
        lora_layer.train(module.training)

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

    # Freeze the base model, then enable the requested adapter/bias parameters.
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if is_lora_module(module):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if config.bias == "lora_only" and module.bias is not None:
                module.bias.requires_grad = True

    if config.bias == "all":
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.requires_grad = True

    return model


def remove_lora(model: nn.Module) -> nn.Module:
    """Remove all LoRA layers, restoring original nn.Linear layers.

    The restored Linear will hold the merged weight if LoRA was merged,
    or the original frozen weight if unmerged. Quantized wrappers
    (``LoRAQuantLinear``) are unwrapped to the original quantized layer
    without merging — 4-bit weights cannot be merged in place.
    """
    modules_to_replace = []
    for name, module in model.named_modules():
        if is_lora_module(module):
            modules_to_replace.append((name, module))

    for name, lora_module in modules_to_replace:
        if isinstance(lora_module, LoRAQuantLinear):
            original = lora_module.base_layer
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            setattr(parent, child_name, original)
            continue

        # Ensure merged before extracting
        if not lora_module.merged and lora_module.r > 0:
            lora_module.merge()

        original = nn.Linear(
            in_features=lora_module.in_features,
            out_features=lora_module.out_features,
            bias=lora_module.bias is not None,
            device=lora_module.weight.device,
            dtype=lora_module.weight.dtype,
        )
        original.weight.data.copy_(lora_module._T(lora_module.weight.data))
        if lora_module.bias is not None:
            original.bias.data.copy_(lora_module.bias.data)
        original.weight.requires_grad = lora_module.weight.requires_grad
        if original.bias is not None and lora_module.bias is not None:
            original.bias.requires_grad = lora_module.bias.requires_grad
        original.train(lora_module.training)

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, original)

    return model
