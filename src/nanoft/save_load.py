"""Adapter save/load utilities."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any, Optional

import torch
import torch.nn as nn

from .apply import apply_lora
from .config import LoRAConfig
from .layers import is_lora_module
from .quant import LoRAQuantLinear
from .merge import merge_and_unload_lora


_PEFT_LORA_KEY = re.compile(
    r"^(?P<module>.*)\.(?P<part>lora_[AB])(?:\.[^.]+)?\.weight$"
)


def _adapter_state_dict(
    model: nn.Module,
    config: LoRAConfig,
) -> dict[str, torch.Tensor]:
    """Extract the LoRA parameters and configured trainable biases."""
    state_dict = model.state_dict()
    adapter_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    if config.bias == "all":
        adapter_state.update(
            (key, value)
            for key, value in state_dict.items()
            if key.endswith(".bias")
        )
    elif config.bias == "lora_only":
        for name, module in model.named_modules():
            if is_lora_module(module) and module.bias is not None:
                # Quantized wrappers keep bias on their base_layer child.
                suffix = ".base_layer.bias" if isinstance(module, LoRAQuantLinear) else ".bias"
                adapter_state[f"{name}{suffix}"] = state_dict[f"{name}{suffix}"]
    return adapter_state


def _validate_adapter_tensors(
    state_dict: dict[str, torch.Tensor],
    expected_state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    """Reject non-tensor, non-finite, or shape-incompatible adapter values."""
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Adapter value for {key!r} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise ValueError(f"Adapter tensor contains non-finite values: {key}")
        if expected_state_dict is not None and key in expected_state_dict:
            expected_shape = expected_state_dict[key].shape
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"Adapter tensor shape mismatch for {key}: "
                    f"expected {tuple(expected_shape)}, got {tuple(value.shape)}"
                )


def _to_peft_key(key: str, lora_module_names: set[str]) -> str:
    """Convert internal state dict key to PEFT naming convention.

    Example:
        model.layers.0.self_attn.q_proj.lora_A
        -> base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    """
    # PEFT strips the runtime adapter name (for example ``default``) when
    # saving adapter weights.
    if key.endswith(".lora_A"):
        base = key[:-7]
        return f"base_model.model.{base}.lora_A.weight"
    elif key.endswith(".lora_B"):
        base = key[:-7]
        return f"base_model.model.{base}.lora_B.weight"
    elif key.endswith(".bias"):
        module_name = key[:-len(".bias")]
        if module_name in lora_module_names:
            key = f"{module_name}.base_layer.bias"
        return f"base_model.model.{key}"
    return key


def _from_peft_key(key: str) -> str:
    """Convert PEFT naming convention back to internal key.

    Only PEFT-prefixed keys are rewritten: native keys of quantized
    wrappers legitimately end in ``.base_layer.bias``.

    PEFT releases before the adapter-weight key format was simplified wrote
    the runtime adapter name (normally ``default``) between the LoRA
    parameter and ``weight``.  Accept both ``lora_A.weight`` and
    ``lora_A.default.weight`` forms when loading.
    """
    if not key.startswith("base_model.model."):
        return key
    key = key[len("base_model.model."):]
    if key.endswith(".base_layer.bias"):
        key = key[:-len(".base_layer.bias")] + ".bias"
    lora_match = _PEFT_LORA_KEY.match(key)
    if lora_match:
        key = f"{lora_match.group('module')}.{lora_match.group('part')}"
    return key


def _save_adapter_state_dict(
    state_dict: dict[str, torch.Tensor],
    save_dir: str,
    safe_serialization: bool,
) -> None:
    if safe_serialization:
        try:
            from safetensors.torch import save_file
            save_file(state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        except ImportError:
            safe_serialization = False

    if not safe_serialization:
        torch.save(state_dict, os.path.join(save_dir, "adapter_model.bin"))


def save_adapter(
    model: nn.Module,
    save_dir: str,
    config: LoRAConfig,
    safe_serialization: bool = True,
    adapter_format: str = "nanoft",
) -> None:
    """Save only LoRA adapter weights + config.

    Writes:
        save_dir/
            adapter_config.json
            adapter_model.safetensors  (or adapter_model.bin)

    Args:
        model: Model containing LoRA layers.
        save_dir: Output directory.
        config: LoRA configuration.
        safe_serialization: Prefer safetensors when available.
        adapter_format: ``"nanoft"`` for native state dict keys, or ``"peft"``
            for PEFT-compatible key names.
    """
    if adapter_format not in {"nanoft", "peft"}:
        raise ValueError("adapter_format must be 'nanoft' or 'peft'")

    # Extract adapter weights
    lora_sd = _adapter_state_dict(model, config)
    if not any("lora_A" in key or "lora_B" in key for key in lora_sd):
        raise ValueError("Model does not contain any LoRA adapter parameters")
    _validate_adapter_tensors(lora_sd)

    os.makedirs(save_dir, exist_ok=True)

    # Save an inference config without mutating the caller's training config.
    replace(config, inference_mode=True).save(
        os.path.join(save_dir, "adapter_config.json")
    )

    if adapter_format == "peft":
        lora_module_names = {
            name for name, module in model.named_modules()
            if is_lora_module(module)
        }
        lora_sd = {
            _to_peft_key(key, lora_module_names): value
            for key, value in lora_sd.items()
        }

    _save_adapter_state_dict(lora_sd, save_dir, safe_serialization)


def save_peft_adapter(
    model: nn.Module,
    save_dir: str,
    config: LoRAConfig,
    safe_serialization: bool = True,
) -> None:
    """Save LoRA adapter weights using PEFT-compatible key names."""
    save_adapter(
        model,
        save_dir,
        config,
        safe_serialization=safe_serialization,
        adapter_format="peft",
    )


def load_adapter(
    model: nn.Module,
    adapter_dir: str,
    device: Optional[str] = None,
    strict: bool = True,
) -> nn.Module:
    """Load LoRA adapter weights into a model.

    Supports NanoFT native and PEFT-compatible adapter weights.
    If the model does not already contain LoRA layers, adapters are injected
    from adapter_config.json before loading weights.
    """
    # Load config
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        config = LoRAConfig.load(config_path)
    else:
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    if not any(is_lora_module(module) for module in model.modules()):
        model = apply_lora(model, config)

    # Load weights
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_sd = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        map_location = device if device else "cpu"
        adapter_sd = torch.load(bin_path, map_location=map_location, weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    # Convert PEFT keys back to internal keys when needed. Bias keys from
    # PEFT exports may target either dense (`.bias`) or quantized-wrapper
    # (`.base_layer.bias`) storage; reconcile against the actual model.
    model_sd = model.state_dict()
    internal_sd = {}
    for k, v in adapter_sd.items():
        internal_key = _from_peft_key(k)
        if internal_key not in model_sd and internal_key.endswith(".bias"):
            wrapper_key = internal_key[: -len(".bias")] + ".base_layer.bias"
            if wrapper_key in model_sd:
                internal_key = wrapper_key
        internal_sd[internal_key] = v
    expected_adapter_sd = _adapter_state_dict(model, config)

    if strict:
        expected_adapter_keys = set(expected_adapter_sd)
        loaded_adapter_keys = set(internal_sd)
        missing_adapter = sorted(expected_adapter_keys - loaded_adapter_keys)
        unexpected_adapter = sorted(loaded_adapter_keys - expected_adapter_keys)
        if missing_adapter or unexpected_adapter:
            details = []
            if missing_adapter:
                details.append(f"missing adapter keys: {missing_adapter}")
            if unexpected_adapter:
                details.append(f"unexpected adapter keys: {unexpected_adapter}")
            raise RuntimeError("Adapter weights do not match model: " + "; ".join(details))

    _validate_adapter_tensors(internal_sd, expected_adapter_sd)

    # Load into model (strict=False to ignore base-model keys absent from adapters).
    result = model.load_state_dict(internal_sd, strict=False)

    if strict:
        loaded_missing_lora = sorted(
            key for key in result.missing_keys
            if "lora_A" in key or "lora_B" in key
        )
        if loaded_missing_lora:
            raise RuntimeError(
                "Adapter weights were not loaded for LoRA keys: "
                + str(loaded_missing_lora)
            )

    return model


def save_merged_model(
    model: nn.Module,
    save_dir: str,
    tokenizer: Optional[Any] = None,
    inplace: bool = False,
) -> None:
    """Save a fully-merged model using HuggingFace format.

    LoRA layers are merged and unloaded before saving, so the saved model can be
    loaded as a normal transformers/PyTorch model without NanoFT LoRA classes.
    By default the input model is preserved; pass ``inplace=True`` to merge and
    unload it destructively with a lower peak-memory cost.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = merge_and_unload_lora(model, inplace=inplace)

    lora_keys = [key for key in model.state_dict() if "lora_A" in key or "lora_B" in key]
    if lora_keys:
        raise RuntimeError(f"Merged model still contains LoRA parameters: {lora_keys}")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)
