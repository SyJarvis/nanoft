"""PEFT-compatible adapter save/load utilities."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from .config import LoRAConfig
from .layers import LoRALinear


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


def save_adapter(
    model: nn.Module,
    save_dir: str,
    config: LoRAConfig,
    safe_serialization: bool = True,
) -> None:
    """Save only LoRA adapter weights + config (PEFT-compatible).

    Writes:
        save_dir/
            adapter_config.json
            adapter_model.safetensors  (or adapter_model.bin)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    config.inference_mode = True
    config.save(os.path.join(save_dir, "adapter_config.json"))

    # Extract and rename LoRA weights
    lora_sd = _lora_state_dict(model)
    peft_sd = {_to_peft_key(k): v for k, v in lora_sd.items()}

    if safe_serialization:
        try:
            from safetensors.torch import save_file
            save_file(peft_sd, os.path.join(save_dir, "adapter_model.safetensors"))
        except ImportError:
            safe_serialization = False

    if not safe_serialization:
        torch.save(peft_sd, os.path.join(save_dir, "adapter_model.bin"))


def load_adapter(
    model: nn.Module,
    adapter_dir: str,
    device: Optional[str] = None,
) -> nn.Module:
    """Load LoRA adapter weights into a model.

    Supports both our format and PEFT format (adapter_config.json + weights).
    """
    # Load config
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        config = LoRAConfig.load(config_path)
    else:
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    # Load weights
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        peft_sd = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        map_location = device if device else "cpu"
        peft_sd = torch.load(bin_path, map_location=map_location, weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    # Convert PEFT keys back to internal keys
    internal_sd = {_from_peft_key(k): v for k, v in peft_sd.items()}

    # Load into model (strict=False to ignore missing/unexpected keys)
    model.load_state_dict(internal_sd, strict=False)

    return model


def save_merged_model(
    model: nn.Module,
    save_dir: str,
    tokenizer: Optional[Any] = None,
) -> None:
    """Save a fully-merged model using HuggingFace format.

    Call merge_lora_weights() before this if LoRA is not yet merged.
    """
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)
