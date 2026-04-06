"""Training preparation utilities for LoRA/QLoRA fine-tuning."""

from __future__ import annotations

import torch.nn as nn

from .apply import apply_lora
from .config import LoRAConfig


def prepare_model_for_training(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Prepare a model for LoRA fine-tuning.

    Steps:
        1. Freeze all existing parameters
        2. Apply LoRA layers to target modules
        3. Verify only LoRA params require gradients

    For QLoRA: call quantize_model_to_nf4() before this function.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA adapters (also handles freezing + LoRA grad enable)
    model = apply_lora(model, config)

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    """Print count and percentage of trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def create_training_args(
    output_dir: str,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
    num_epochs: int = 3,
    bf16: bool = False,
    fp16: bool = False,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    save_total_limit: int = 2,
    **kwargs,
):
    """Create HuggingFace TrainingArguments with sensible LoRA defaults."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        bf16=bf16,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        **kwargs,
    )
