"""QLoRA logic tests using a pure-PyTorch fake-quantization stub.

These tests run everywhere (no bitsandbytes/CUDA required): a
``FakeQuantLinear`` holds a non-Tensor ``FakeParams`` weight that simulates
a 4-bit quantize->dequantize round-trip, duck-typing the same contract as
``bnb.nn.Linear4bit`` / ``Params4bit``.
"""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoft import (
    LoRAConfig,
    LoRALinear,
    QuantizationConfig,
    QLoRAConfig,
    apply_lora,
    dequantize_and_merge,
    is_quantized_linear,
    load_adapter,
    merge_and_unload_lora,
    merge_lora_weights,
    remove_lora,
    save_adapter,
    save_peft_adapter,
    unmerge_lora_weights,
)
from nanoft.layers import is_lora_module
from nanoft.quant import LoRAQuantLinear


class FakeParams:
    """Non-Tensor stand-in for a quantized weight (Params4bit duck-type)."""

    def __init__(self, weight: torch.Tensor, bits: int = 4):
        self._dense = weight.detach().clone()
        self._bits = bits
        self.quant_state = None
        self.device = weight.device

    def dequantize(self) -> torch.Tensor:
        """Simulate dequantizing from a 2^bits uniform grid."""
        levels = 2 ** self._bits
        t_min = self._dense.min()
        scale = (self._dense.max() - t_min) / (levels - 1)
        if scale == 0:
            return self._dense.clone()
        q = torch.round((self._dense - t_min) / scale)
        return q * scale + t_min


class FakeQuantLinear(nn.Linear):
    """nn.Linear whose weight is replaced by a non-Tensor FakeParams."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        dense = self.weight.data
        # Remove the registered Parameter so state_dict()/parameters() skip
        # the fake weight entirely, mirroring a 4-bit storage layer.
        del self._parameters["weight"]
        self.weight = FakeParams(dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.dequantize(), self.bias)


class FakeQuantBlock(nn.Module):
    def __init__(self, d: int = 8, bias: bool = False):
        super().__init__()
        self.q_proj = FakeQuantLinear(d, d, bias=bias)
        self.v_proj = FakeQuantLinear(d, d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.v_proj(x)


class DenseBlock(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def _fake_model_with_lora(config=None, seed=0):
    torch.manual_seed(seed)
    model = FakeQuantBlock()
    config = config or LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    return apply_lora(model, config), config


class TestPredicate:
    def test_fake_quant_linear_is_quantized(self):
        assert is_quantized_linear(FakeQuantLinear(4, 4))

    def test_dense_linear_is_not_quantized(self):
        assert not is_quantized_linear(nn.Linear(4, 4))

    def test_lora_linear_is_not_quantized(self):
        assert not is_quantized_linear(LoRALinear(4, 4, r=2, lora_alpha=4))

    def test_wrapper_is_not_quantized_linear(self):
        model, _ = _fake_model_with_lora()
        assert not is_quantized_linear(model.q_proj)
        assert isinstance(model.q_proj, LoRAQuantLinear)
        assert is_lora_module(model.q_proj)
        assert is_lora_module(LoRALinear(4, 4, r=2, lora_alpha=4))
        assert not is_lora_module(nn.Linear(4, 4))


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"backend": "torchao"}, "bnb"),
            ({"quant_type": "int8"}, "nf4"),
            ({"compute_dtype": "int8"}, "float16"),
            ({"exclude_modules": ("",)}, "exclude_modules"),
        ],
    )
    def test_invalid_quantization_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            QuantizationConfig(**kwargs)

    def test_qlora_config_default_quantization_independent(self):
        a = QLoRAConfig(lora=LoRAConfig())
        b = QLoRAConfig(lora=LoRAConfig())
        assert a.quantization is not b.quantization
        assert a.quantization.backend == "bnb"
        assert a.quantization.quant_type == "nf4"


class TestQuantizeModelGuards:
    def test_requires_cuda(self, monkeypatch):
        from nanoft import quantize_model

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA"):
            quantize_model(nn.Linear(4, 4), QuantizationConfig())

    def test_requires_bitsandbytes(self, monkeypatch):
        from nanoft import quantize_model

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)
        with pytest.raises(RuntimeError, match="bitsandbytes"):
            quantize_model(nn.Linear(4, 4), QuantizationConfig())


class TestApplyLoraQuantized:
    def test_wraps_quantized_targets(self):
        model, _ = _fake_model_with_lora()
        assert isinstance(model.q_proj, LoRAQuantLinear)
        assert isinstance(model.q_proj.base_layer, FakeQuantLinear)
        # v_proj untouched
        assert isinstance(model.v_proj, FakeQuantLinear)

    def test_repeated_injection_rejected(self):
        model, _ = _fake_model_with_lora()
        with pytest.raises(ValueError, match="repeated injection"):
            apply_lora(model, LoRAConfig(r=2, lora_alpha=4))

    def test_lora_params_trainable_base_frozen(self):
        model, _ = _fake_model_with_lora()
        trainable = {
            n for n, p in model.named_parameters()
            if p.requires_grad and "lora" not in n
        }
        assert trainable == set()
        assert model.q_proj.lora_A.requires_grad
        assert model.q_proj.lora_B.requires_grad

    def test_forward_matches_reference(self):
        model, _ = _fake_model_with_lora()
        wrapper = model.q_proj
        with torch.no_grad():
            wrapper.lora_A.normal_()
            wrapper.lora_B.normal_()
        model.eval()
        x = torch.randn(3, 8)

        w = wrapper.base_layer.weight.dequantize()
        expected = F.linear(x, w) + (
            (x @ wrapper.lora_A.T @ wrapper.lora_B.T) * wrapper.scaling
        )
        assert torch.allclose(model.q_proj(x), expected, atol=1e-6)

    def test_gradients_flow_only_to_adapters(self):
        model, _ = _fake_model_with_lora()
        with torch.no_grad():
            model.q_proj.lora_A.normal_()
            model.q_proj.lora_B.normal_()
        x = torch.randn(3, 8)
        model(x).sum().backward()

        assert model.q_proj.lora_A.grad is not None
        assert model.q_proj.lora_B.grad is not None
        grads_outside = [
            n for n, p in model.named_parameters()
            if p.grad is not None and "lora" not in n
        ]
        assert grads_outside == []
        # The fake quantized weight was not silently converted to a tensor.
        assert not isinstance(model.q_proj.base_layer.weight, torch.Tensor)


class TestMergeSemantics:
    @pytest.mark.parametrize(
        "merge_fn",
        [merge_lora_weights, unmerge_lora_weights, merge_and_unload_lora],
    )
    def test_dense_merge_apis_reject_quantized(self, merge_fn):
        model, _ = _fake_model_with_lora()
        with pytest.raises(TypeError, match="dequantize_and_merge"):
            merge_fn(model)

    def test_wrapper_merge_raises(self):
        model, _ = _fake_model_with_lora()
        with pytest.raises(TypeError, match="dequantize_and_merge"):
            model.q_proj.merge()
        with pytest.raises(TypeError, match="never merged in place"):
            model.q_proj.unmerge()

    def test_dequantize_and_merge_matches_dense_reference(self):
        model, _ = _fake_model_with_lora()
        wrapper = model.q_proj
        with torch.no_grad():
            wrapper.lora_A.normal_()
            wrapper.lora_B.normal_()
        model.eval()
        x = torch.randn(3, 8)
        expected = model.q_proj(x)

        merged = dequantize_and_merge(model)

        assert isinstance(merged.q_proj, nn.Linear)
        assert not any(is_lora_module(m) for m in merged.modules())
        assert not any("lora_A" in k or "lora_B" in k for k in merged.state_dict())
        w_ref = wrapper.base_layer.weight.dequantize() + (
            wrapper.lora_B @ wrapper.lora_A
        ) * wrapper.scaling
        assert torch.allclose(merged.q_proj.weight, w_ref, atol=1e-6)
        assert torch.allclose(merged.q_proj(x), expected, atol=1e-6)

    def test_dequantize_and_merge_non_destructive(self):
        model, _ = _fake_model_with_lora()
        merged = dequantize_and_merge(model)
        assert isinstance(merged.q_proj, nn.Linear)
        assert isinstance(model.q_proj, LoRAQuantLinear)

    def test_dequantize_rejects_dense_only_model(self):
        torch.manual_seed(0)
        dense = apply_lora(
            DenseBlock(),
            LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"]),
        )
        with pytest.raises(ValueError, match="merge_and_unload_lora"):
            dequantize_and_merge(dense)

    def test_dequantize_rejects_mixed_quantized_and_dense_lora(self):
        model = FakeQuantBlock()
        model.v_proj = nn.Linear(8, 8)
        config = LoRAConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
        )
        model = apply_lora(model, config)

        with pytest.raises(ValueError, match="both quantized and dense"):
            dequantize_and_merge(model)

    def test_remove_lora_unwraps_original_module(self):
        model, _ = _fake_model_with_lora()
        original = model.q_proj.base_layer
        restored = remove_lora(model)
        assert restored.q_proj is original
        assert not any(is_lora_module(m) for m in restored.modules())


class TestSaveLoad:
    def _fill_adapter(self, model, value_a, value_b):
        with torch.no_grad():
            model.q_proj.lora_A.fill_(value_a)
            model.q_proj.lora_B.fill_(value_b)

    def test_native_round_trip(self, tmp_path):
        model, config = _fake_model_with_lora()
        self._fill_adapter(model, 0.25, 0.5)
        model.eval()
        x = torch.randn(3, 8)
        expected = model(x)

        save_adapter(model, str(tmp_path), config)
        fresh = FakeQuantBlock()
        # FakeParams is deliberately excluded from state_dict(), so preserve
        # the base weights explicitly before loading the adapter-only file.
        fresh.q_proj.weight._dense.copy_(model.q_proj.base_layer.weight._dense)
        fresh.v_proj.weight._dense.copy_(model.v_proj.weight._dense)
        loaded = load_adapter(fresh, str(tmp_path))
        loaded.eval()

        assert isinstance(loaded.q_proj, LoRAQuantLinear)
        assert torch.allclose(loaded(x), expected, atol=1e-6)

    def test_peft_export_key_format(self, tmp_path):
        model, config = _fake_model_with_lora()
        self._fill_adapter(model, 0.25, 0.5)
        save_peft_adapter(model, str(tmp_path), config)

        from safetensors.torch import load_file

        saved = load_file(str(tmp_path / "adapter_model.safetensors"))
        assert "base_model.model.q_proj.lora_A.weight" in saved
        assert "base_model.model.q_proj.lora_B.weight" in saved
        assert torch.allclose(saved["base_model.model.q_proj.lora_A.weight"], torch.full((2, 8), 0.25))

    def test_bias_lora_only_includes_base_layer_bias(self, tmp_path):
        torch.manual_seed(0)
        model = FakeQuantBlock(bias=True)
        config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"], bias="lora_only")
        model = apply_lora(model, config)
        with torch.no_grad():
            model.q_proj.base_layer.bias.fill_(0.25)

        save_adapter(model, str(tmp_path), config)

        from safetensors.torch import load_file

        saved = load_file(str(tmp_path / "adapter_model.safetensors"))
        assert "q_proj.base_layer.bias" in saved
        assert "q_proj.lora_A" in saved

        # Round-trip restores bias and adapter weights on a fresh model.
        fresh = apply_lora(FakeQuantBlock(bias=True), config)
        loaded = load_adapter(fresh, str(tmp_path))
        assert torch.allclose(loaded.q_proj.base_layer.bias, torch.full((8,), 0.25))

    def test_peft_export_bias_key_format(self, tmp_path):
        torch.manual_seed(0)
        model = FakeQuantBlock(bias=True)
        config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"], bias="lora_only")
        model = apply_lora(model, config)
        save_peft_adapter(model, str(tmp_path), config)

        from safetensors.torch import load_file

        saved = load_file(str(tmp_path / "adapter_model.safetensors"))
        assert "base_model.model.q_proj.base_layer.bias" in saved


class TestPrepareModelForTraining:
    def test_qlora_config_routes_through_quantize(self, monkeypatch):
        from nanoft import prepare_model_for_training
        from nanoft import train_utils

        calls = {}

        def fake_quantize(model, config, device=None):
            calls["quantize"] = True
            return model

        monkeypatch.setattr(train_utils, "quantize_model", fake_quantize)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        model = FakeQuantBlock()
        qlora = QLoRAConfig(
            lora=LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"]),
            quantization=QuantizationConfig(),
        )
        out = prepare_model_for_training(model, qlora)

        assert calls.get("quantize") is True
        assert isinstance(out.q_proj, LoRAQuantLinear)
        assert out.q_proj.lora_A.dtype == torch.bfloat16
