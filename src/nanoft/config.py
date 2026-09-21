"""LoRA / QLoRA configuration with PEFT-compatible serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class LoRAConfig:
    """Configuration for LoRA / QLoRA fine-tuning.

    Serializable to/from PEFT's adapter_config.json format.
    """

    # Core LoRA parameters
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = field(default=None)
    fan_in_fan_out: bool = False
    bias: str = "none"  # "none" | "all" | "lora_only"

    # PEFT compatibility fields
    task_type: str = "CAUSAL_LM"
    base_model_name_or_path: Optional[str] = None
    inference_mode: bool = False
    peft_type: str = "LORA"

    # Default target modules when none specified
    _DEFAULT_TARGETS: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"],
        repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        """Validate the LoRA behavior NanoFT currently implements."""
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r <= 0:
            raise ValueError("r must be a positive integer")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be greater than 0")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in the range [0, 1)")
        if self.target_modules is not None and not self.target_modules:
            raise ValueError("target_modules must be None or a non-empty list")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias must be 'none', 'all', or 'lora_only'")
        if self.fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out is not supported for NanoFT nn.Linear adapters"
            )

    def get_target_modules(self) -> List[str]:
        """Return effective target modules (user-specified or defaults)."""
        return self.target_modules if self.target_modules else list(self._DEFAULT_TARGETS)

    def to_peft_dict(self) -> dict:
        """Serialize to PEFT-compatible adapter_config.json dict."""
        d = asdict(self)
        # Add PEFT-required fields
        d["auto_mapping"] = None
        d["revision"] = None
        d["modules_to_save"] = None
        d["init_lora_weights"] = True
        d["layers_to_transform"] = None
        d["layers_pattern"] = None
        d["rank_pattern"] = {}
        d["alpha_pattern"] = {}
        d.pop("_DEFAULT_TARGETS", None)
        return d

    @classmethod
    def from_peft_dict(cls, d: dict) -> LoRAConfig:
        """Deserialize from PEFT adapter_config.json dict."""
        return cls(
            r=d.get("r", 8),
            lora_alpha=d.get("lora_alpha", 16),
            lora_dropout=d.get("lora_dropout", 0.0),
            target_modules=d.get("target_modules"),
            fan_in_fan_out=d.get("fan_in_fan_out", False),
            bias=d.get("bias", "none"),
            task_type=d.get("task_type", "CAUSAL_LM"),
            base_model_name_or_path=d.get("base_model_name_or_path"),
            inference_mode=d.get("inference_mode", False),
        )

    def save(self, path: str) -> None:
        """Write adapter_config.json to *path*."""
        with open(path, "w") as f:
            json.dump(self.to_peft_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> LoRAConfig:
        """Read adapter_config.json from *path*."""
        with open(path) as f:
            return cls.from_peft_dict(json.load(f))


@dataclass
class QuantizationConfig:
    """Configuration for quantizing base weights before LoRA injection (QLoRA).

    Only the bitsandbytes 4-bit backend is supported. Quantization runs on
    CUDA; see ``nanoft.quant.quantize_model`` for the runtime requirements.
    """

    backend: str = "bnb"
    quant_type: str = "nf4"  # "nf4" | "fp4"
    compute_dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"
    use_double_quant: bool = True
    exclude_modules: Tuple[str, ...] = ("lm_head",)

    def __post_init__(self) -> None:
        """Validate the backends NanoFT currently implements."""
        if self.backend != "bnb":
            raise ValueError(
                "QuantizationConfig only supports backend='bnb' (bitsandbytes)"
            )
        if self.quant_type not in {"nf4", "fp4"}:
            raise ValueError("quant_type must be 'nf4' or 'fp4'")
        if self.compute_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "compute_dtype must be 'float16', 'bfloat16', or 'float32'"
            )
        if not all(isinstance(m, str) and m for m in self.exclude_modules):
            raise ValueError("exclude_modules must contain non-empty strings")


@dataclass
class QLoRAConfig:
    """Composition of a LoRA adapter config with base-weight quantization.

    ``quantize_model(model, config.quantization)`` freezes the base weights in
    4-bit; ``apply_lora(model, config.lora)`` then trains full-precision LoRA
    adapters on top. ``prepare_model_for_training`` accepts this config and
    performs both steps in order.
    """

    lora: LoRAConfig
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
