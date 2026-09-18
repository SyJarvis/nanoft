"""NanoFT: lightweight edge-side LLM fine-tuning framework and toolkit."""

__version__ = "0.2.0"

from .config import LoRAConfig, QuantizationConfig, QLoRAConfig
from .layers import LoRALinear
from .apply import apply_lora, remove_lora
from .merge import merge_lora_weights, unmerge_lora_weights, merge_and_unload_lora
from .quant import (
    LoRAQuantLinear,
    dequantize_and_merge,
    is_quantized_linear,
    quantize_model,
)
from .save_load import save_adapter, save_peft_adapter, load_adapter, save_merged_model
from .device import (
    DeviceInfo,
    detect_device,
    get_recommended_dtype,
    move_to_device,
)
from .train_utils import prepare_model_for_training, print_trainable_parameters

__all__ = [
    "DeviceInfo",
    "LoRAConfig",
    "LoRAQuantLinear",
    "LoRALinear",
    "QLoRAConfig",
    "QuantizationConfig",
    "apply_lora",
    "dequantize_and_merge",
    "detect_device",
    "get_recommended_dtype",
    "is_quantized_linear",
    "load_adapter",
    "merge_and_unload_lora",
    "merge_lora_weights",
    "move_to_device",
    "prepare_model_for_training",
    "print_trainable_parameters",
    "quantize_model",
    "remove_lora",
    "save_adapter",
    "save_merged_model",
    "save_peft_adapter",
    "unmerge_lora_weights",
]
