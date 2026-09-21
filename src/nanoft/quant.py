"""QLoRA support: 4-bit quantized bases with full-precision LoRA adapters.

Scheme: the base weights stay frozen in 4-bit (NF4/FP4 via bitsandbytes);
LoRA ``lora_A`` / ``lora_B`` are trained in the compute dtype. Forward:

    output = base_layer(x) + (B @ A(dropout(x))) * scaling

The quantized weight is never copied or mutated: dense ``merge()`` is
impossible on 4-bit weights, so ``dequantize_and_merge()`` is the only
merge path for quantized models.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .config import QuantizationConfig
from .layers import LoRALayer

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def is_quantized_weight(weight: object) -> bool:
    """Duck-type a 4-bit weight (works without importing bitsandbytes).

    bitsandbytes ``Params4bit`` is a ``Parameter`` subclass distinguishable
    by its ``quant_state`` attribute; test stubs hold non-Tensor weights.
    """
    return hasattr(weight, "quant_state") or not isinstance(weight, torch.Tensor)


def is_quantized_linear(module: nn.Module) -> bool:
    """Return True if *module* is a Linear whose weight is 4-bit quantized."""
    if not isinstance(module, nn.Linear):
        return False
    return is_quantized_weight(getattr(module, "weight", None))


def _weight_device(weight: object) -> torch.device:
    device = getattr(weight, "device", None)
    return device if isinstance(device, torch.device) else torch.device("cpu")


def _compute_dtype(quantization: QuantizationConfig) -> torch.dtype:
    return _DTYPE_MAP[quantization.compute_dtype]


class LoRAQuantLinear(nn.Module, LoRALayer):
    """LoRA adapter composed around a quantized Linear base layer.

    The base layer is kept as the ``base_layer`` child module, untouched —
    its 4-bit weight is never copied or merged in place. ``lora_A`` /
    ``lora_B`` live directly on this wrapper in the compute dtype, so
    adapter state-dict keys match the dense ``LoRALinear`` naming
    (``...lora_A`` / ``...lora_B``) and PEFT export works unchanged.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(
            self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=False,
        )
        self.base_layer = base_layer
        dtype = compute_dtype if compute_dtype is not None else torch.get_default_dtype()
        device = _weight_device(base_layer.weight)
        self.lora_A = nn.Parameter(
            torch.empty(r, base_layer.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(base_layer.out_features, r, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def bias(self) -> Optional[nn.Parameter]:
        return self.base_layer.bias

    def merge(self) -> None:
        raise TypeError(
            "cannot merge LoRA weights into 4-bit quantized weights; "
            "use nanoft.quant.dequantize_and_merge()"
        )

    def unmerge(self) -> None:
        raise TypeError(
            "cannot unmerge LoRA weights from 4-bit quantized weights; "
            "quantized bases are never merged in place"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.r > 0 and not self.merged:
            lora_out = (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
            result = result + lora_out
        return result

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, lora_alpha={self.lora_alpha}, "
            f"merged={self.merged}, base_layer={self.base_layer.extra_repr()}"
        )


def _module_is_excluded(name: str, exclude_modules: tuple) -> bool:
    return any(name == m or name.endswith(f".{m}") for m in exclude_modules)


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    device: Optional[str] = None,
) -> nn.Module:
    """Quantize a loaded model's Linear weights to 4-bit (in-place).

    Replaces every ``nn.Linear`` (except ``config.exclude_modules`` matches,
    e.g. ``lm_head``) with ``bnb.nn.Linear4bit`` and moves the model to
    *device* — bitsandbytes quantizes lazily when ``Params4bit`` reaches
    the GPU, so the final ``.to(device)`` triggers quantization. Linears
    that are already quantized are left untouched, so this is safe on
    models loaded with ``BitsAndBytesConfig``.

    Requires CUDA and bitsandbytes; MPS is not supported for quantized
    training (see docs/getting-started/device-support.md).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA quantization with backend 'bnb' requires CUDA; "
            "no CUDA device is available"
        )
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise RuntimeError(
            "bitsandbytes is required for QLoRA; "
            "install it with: pip install 'nanoft[qlora]'"
        ) from e

    compute_dtype = _compute_dtype(config)
    replacements = []
    for name, module in model.named_modules():
        if _module_is_excluded(name, config.exclude_modules):
            continue
        if isinstance(module, nn.Linear) and not is_quantized_linear(module):
            replacements.append((name, module))

    for name, module in replacements:
        quant = bnb.nn.Linear4bit(
            input_features=module.in_features,
            output_features=module.out_features,
            bias=module.bias is not None,
            compute_dtype=compute_dtype,
            quant_type=config.quant_type,
            compress_statistics=config.use_double_quant,
        )
        quant.weight = type(quant.weight)(
            module.weight.data,
            requires_grad=False,
            quant_type=config.quant_type,
            compress_statistics=config.use_double_quant,
        )
        if module.bias is not None:
            quant.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        quant.train(module.training)

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, quant)

    model.to(device if device is not None else "cuda")
    return model


def _dequantize_weight(weight: object) -> torch.Tensor:
    """Return the dense weight tensor from a possibly-quantized weight."""
    if hasattr(weight, "dequantize") and callable(weight.dequantize):
        result = weight.dequantize()
        if isinstance(result, torch.Tensor):
            return result
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        import bitsandbytes as bnb

        return bnb.functional.dequantize_4bit(weight, quant_state)
    if isinstance(weight, torch.Tensor):
        return weight
    raise TypeError(
        f"cannot dequantize weight of type {type(weight).__name__}"
    )


def dequantize_and_merge(
    model: nn.Module,
    dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
) -> nn.Module:
    """Merge LoRA deltas into dequantized weights; return a standard model.

    For every ``LoRAQuantLinear``: dequantize the 4-bit base to
    ``dtype`` (default: the weight's compute dtype), add
    ``(B @ A) * scaling``, and swap in a plain ``nn.Linear``. The returned
    model has no quantized or LoRA modules and can feed
    ``save_merged_model``. The input model is preserved unless
    ``inplace=True``.
    """
    if not inplace:
        model = copy.deepcopy(model)

    from .layers import LoRALinear

    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, LoRAQuantLinear):
            modules_to_replace.append((name, module))
    dense_lora = [
        name for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if dense_lora:
        if modules_to_replace:
            raise ValueError(
                "cannot dequantize a model containing both quantized and "
                "dense LoRA layers; use merge_and_unload_lora() for dense "
                "layers first: " + ", ".join(dense_lora)
            )
        raise ValueError(
            "no quantized LoRA layers found; dense LoRALinear modules "
            "must use merge_and_unload_lora(): " + ", ".join(dense_lora)
        )

    for name, wrapper in modules_to_replace:
        base = wrapper.base_layer
        weight = _dequantize_weight(base.weight).to(
            dtype if dtype is not None else torch.get_default_dtype()
        )
        if wrapper.r > 0:
            delta = (wrapper.lora_B.to(weight.dtype) @ wrapper.lora_A.to(weight.dtype))
            weight = weight + delta * wrapper.scaling

        linear = nn.Linear(
            in_features=base.in_features,
            out_features=base.out_features,
            bias=base.bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        linear.weight.data.copy_(weight)
        if base.bias is not None:
            linear.bias.data.copy_(base.bias.data.to(weight.dtype))
        linear.weight.requires_grad = False
        linear.train(wrapper.training)

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
        setattr(parent, child_name, linear)

    return model
