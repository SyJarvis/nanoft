"""Adapter save/load utilities."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from .apply import apply_lora
from .config import LoRAConfig
from .layers import LoRALinear
from .merge import merge_and_unload_lora


def _lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model state dict."""
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def _to_peft_key(key: str) -> str:
    """Convert internal state dict key to PEFT naming convention.

    Example:
        model.layers.0.self_attn.q_proj.lora_A
        -> base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    """
    # Map lora_A / lora_B -> lora_A.default.weight / lora_B.default.weight
    if key.endswith(".lora_A"):
        base = key[:-7]
        return f"base_model.model.{base}.lora_A.default.weight"
    elif key.endswith(".lora_B"):
        base = key[:-7]
        return f"base_model.model.{base}.lora_B.default.weight"
    return key


def _from_peft_key(key: str) -> str:
    """Convert PEFT naming convention back to internal key."""
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]
    key = key.replace(".lora_A.default.weight", ".lora_A")
    key = key.replace(".lora_B.default.weight", ".lora_B")
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

    os.makedirs(save_dir, exist_ok=True)

    # Save config
    config.inference_mode = True
    config.save(os.path.join(save_dir, "adapter_config.json"))

    # Extract LoRA weights
    lora_sd = _lora_state_dict(model)
    if adapter_format == "peft":
        lora_sd = {_to_peft_key(k): v for k, v in lora_sd.items()}

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

    if not any(isinstance(module, LoRALinear) for module in model.modules()):
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

    # Convert PEFT keys back to internal keys when needed.
    internal_sd = {_from_peft_key(k): v for k, v in adapter_sd.items()}

    # Load into model (strict=False to ignore missing/unexpected keys)
    result = model.load_state_dict(internal_sd, strict=False)

    if strict:
        expected_lora_keys = {
            key for key in model.state_dict()
            if "lora_A" in key or "lora_B" in key
        }
        loaded_lora_keys = {
            key for key in internal_sd
            if "lora_A" in key or "lora_B" in key
        }
        missing_lora = sorted(expected_lora_keys - loaded_lora_keys)
        unexpected_lora = sorted(loaded_lora_keys - expected_lora_keys)
        if missing_lora or unexpected_lora:
            details = []
            if missing_lora:
                details.append(f"missing LoRA keys: {missing_lora}")
            if unexpected_lora:
                details.append(f"unexpected LoRA keys: {unexpected_lora}")
            raise RuntimeError("Adapter weights do not match model: " + "; ".join(details))

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
    inplace: bool = True,
) -> None:
    """Save a fully-merged model using HuggingFace format.

    LoRA layers are merged and unloaded before saving, so the saved model can be
    loaded as a normal transformers/PyTorch model without NanoFT LoRA classes.
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
