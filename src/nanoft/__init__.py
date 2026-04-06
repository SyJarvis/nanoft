"""NanoFT: Nano Fine-Tuning — lightweight edge-side LLM fine-tuning framework."""

__version__ = "0.1.0"

from .config import LoRAConfig
from .layers import LoRALinear
from .apply import apply_lora, remove_lora
from .merge import merge_lora_weights, unmerge_lora_weights
from .save_load import save_adapter, load_adapter, save_merged_model
from .device import detect_device
from .train_utils import prepare_model_for_training, print_trainable_parameters
