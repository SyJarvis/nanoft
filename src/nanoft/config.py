"""LoRA configuration with PEFT-compatible serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


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

    # QLoRA parameters (internal, stripped from PEFT output)
    quantize_base: bool = False
    compute_dtype: str = "float16"
    double_quant: bool = False
    quant_storage_dtype: str = "float32"

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
        # Strip internal QLoRA fields
        for k in ("quantize_base", "compute_dtype", "double_quant",
                   "quant_storage_dtype", "_DEFAULT_TARGETS"):
            d.pop(k, None)
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
