"""LoRA layer implementations — refactored from model.py."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer:
    """Mixin providing LoRA state (rank, alpha, dropout, merge state)."""

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )
        self.merged = False
        self.merge_weights = merge_weights

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r


class LoRALinear(nn.Linear, LoRALayer):
    """Drop-in replacement for nn.Linear with LoRA adapters.

    Forward (unmerged):
        output = F.linear(x, W, b) + (dropout(x) @ A^T @ B^T) * scaling

    Forward (merged):
        output = F.linear(x, W + B @ A * scaling, b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        # Freeze original weight
        self.weight.requires_grad = False

        self.reset_lora_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_lora_parameters(self) -> None:
        """Initialize LoRA parameters: A with kaiming uniform, B with zeros."""
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _T(self, w: torch.Tensor) -> torch.Tensor:
        """Transpose weight if fan_in_fan_out mode."""
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge(self) -> None:
        """Merge LoRA delta into base weight in-place."""
        if self.merged or self.r == 0:
            return
        self.weight.data += self._T(self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge LoRA delta from base weight in-place."""
        if not self.merged or self.r == 0:
            return
        self.weight.data -= self._T(self.lora_B @ self.lora_A) * self.scaling
        self.merged = False

    def train(self, mode: bool = True):
        """Switch between train/eval, managing merge state."""
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                self.unmerge()
        else:
            if self.merge_weights and not self.merged:
                self.merge()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional LoRA adapter."""
        result = F.linear(x, self._T(self.weight), bias=self.bias)
        if self.r > 0 and not self.merged:
            lora_out = (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            result = result + lora_out
        return result

    def extra_repr(self) -> str:
        base = nn.Linear.extra_repr(self)
        if self.r > 0:
            base += f", r={self.r}, lora_alpha={self.lora_alpha}, merged={self.merged}"
        return base
