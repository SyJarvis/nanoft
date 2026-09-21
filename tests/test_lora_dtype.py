import copy
import json

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from nanoft import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    load_adapter,
    merge_and_unload_lora,
    prepare_model_for_training,
    save_adapter,
)


class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)

    def forward(self, x):
        return self.q_proj(x)


def _config():
    return LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])


def test_dense_adapter_dtype_is_opt_in():
    default = prepare_model_for_training(
        ProjectionModel().to(torch.bfloat16), _config()
    )
    explicit = prepare_model_for_training(
        ProjectionModel().to(torch.bfloat16), _config(),
        adapter_dtype=torch.float32,
    )
    assert default.q_proj.lora_A.dtype == torch.bfloat16
    assert default.q_proj.lora_B.dtype == torch.bfloat16
    assert explicit.q_proj.weight.dtype == torch.bfloat16
    assert explicit.q_proj.bias.dtype == torch.bfloat16
    assert explicit.q_proj.lora_A.dtype == torch.float32
    assert explicit.q_proj.lora_B.dtype == torch.float32
    assert not torch.equal(
        explicit.q_proj.lora_A,
        explicit.q_proj.lora_A.to(torch.bfloat16).float(),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("autocast", [False, True])
def test_fp32_adapters_match_peft_forward_backward_and_adamw(device, autocast):
    peft = pytest.importorskip("peft")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(42)
    base = ProjectionModel().to(device=device, dtype=torch.bfloat16)
    base_state = copy.deepcopy(base.state_dict())
    native = apply_lora(copy.deepcopy(base), _config(), adapter_dtype=torch.float32)
    reference = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]),
    )
    native_layer = native.q_proj
    peft_layer = reference.base_model.model.q_proj
    with torch.no_grad():
        native_layer.lora_A.normal_(std=0.1)
        native_layer.lora_B.normal_(std=0.1)
        peft_layer.lora_A["default"].weight.copy_(native_layer.lora_A)
        peft_layer.lora_B["default"].weight.copy_(native_layer.lora_B)
    pairs = [
        (native_layer.lora_A, peft_layer.lora_A["default"].weight),
        (native_layer.lora_B, peft_layer.lora_B["default"].weight),
    ]
    inputs = torch.randn(3, 8, device=device, dtype=torch.bfloat16)
    target = torch.randn(3, 8, device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
        actual = native(inputs)
        expected = reference(inputs)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual.float() - target).square().mean().backward()
    (expected.float() - target).square().mean().backward()

    for native_param, peft_param in pairs:
        assert native_param.dtype == peft_param.dtype == torch.float32
        assert native_param.grad.dtype == peft_param.grad.dtype == torch.float32
        assert torch.isfinite(native_param.grad).all()
        assert torch.count_nonzero(native_param.grad) > 0
        torch.testing.assert_close(native_param.grad, peft_param.grad, rtol=0, atol=0)

    native_optimizer = torch.optim.AdamW(
        [p for p in native.parameters() if p.requires_grad], lr=2e-4,
        foreach=False,
    )
    peft_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=2e-4,
        foreach=False,
    )
    native_optimizer.step()
    peft_optimizer.step()
    for native_param, peft_param in pairs:
        torch.testing.assert_close(native_param, peft_param, rtol=0, atol=0)
        state = native_optimizer.state[native_param]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state[key], peft_optimizer.state[peft_param][key], rtol=0, atol=0
            )
    for name, param in native.named_parameters():
        if "lora_" not in name:
            assert not param.requires_grad
            assert param.grad is None
            assert torch.equal(param, base_state[name])


@pytest.mark.parametrize("adapter_format", ["nanoft", "peft"])
@pytest.mark.parametrize("preinjected", [False, True])
def test_fp32_adapter_round_trip_preserves_values(tmp_path, adapter_format, preinjected):
    base = ProjectionModel().to(torch.bfloat16)
    trained = apply_lora(copy.deepcopy(base), _config(), adapter_dtype=torch.float32)
    with torch.no_grad():
        trained.q_proj.lora_A.fill_(0.123456789)
        trained.q_proj.lora_B.fill_(0.234567891)
    save_adapter(trained, str(tmp_path), _config(), adapter_format=adapter_format)
    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert all(value.dtype == torch.float32 for value in saved.values())
    assert "adapter_dtype" not in json.loads(
        (tmp_path / "adapter_config.json").read_text()
    )
    if preinjected:
        base = apply_lora(base, _config())
        assert base.q_proj.lora_A.dtype == torch.bfloat16
    restored = load_adapter(base, str(tmp_path), adapter_dtype=torch.float32)
    for key in ("lora_A", "lora_B"):
        expected = getattr(trained.q_proj, key)
        actual = getattr(restored.q_proj, key)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        assert not torch.equal(actual, actual.to(torch.bfloat16).float())
    inputs = torch.randn(3, 8, dtype=torch.bfloat16)
    assert torch.equal(trained(inputs), restored(inputs))
    assert restored.q_proj.weight.dtype == torch.bfloat16


def test_mixed_dtype_merge_matches_peft_default_merge():
    peft = pytest.importorskip("peft")
    torch.manual_seed(42)
    base = ProjectionModel().to(torch.bfloat16)
    native = apply_lora(copy.deepcopy(base), _config(), adapter_dtype=torch.float32)
    reference = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]),
    )
    with torch.no_grad():
        native.q_proj.lora_A.normal_(std=0.2)
        native.q_proj.lora_B.normal_(std=0.2)
        peft_layer = reference.base_model.model.q_proj
        peft_layer.lora_A["default"].weight.copy_(native.q_proj.lora_A)
        peft_layer.lora_B["default"].weight.copy_(native.q_proj.lora_B)
    base_weight = native.q_proj.weight.detach().clone()
    delta = native.q_proj.lora_B @ native.q_proj.lora_A * native.q_proj.scaling
    expected_weight = (base_weight.float() + delta).to(torch.bfloat16)
    merged = merge_and_unload_lora(native, inplace=False)
    peft_merged = reference.merge_and_unload()
    assert torch.equal(merged.q_proj.weight, expected_weight)
    assert torch.equal(merged.q_proj.weight, peft_merged.q_proj.weight)
    assert merged.q_proj.weight.dtype == torch.bfloat16
    assert isinstance(native.q_proj, LoRALinear)
    assert torch.equal(native.q_proj.weight, base_weight)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16)
    assert torch.equal(merged(inputs), peft_merged(inputs))


def test_fp32_adapter_export_reloads_with_peft(tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )).to(torch.bfloat16)
    trained = apply_lora(copy.deepcopy(base), _config(), adapter_dtype=torch.float32)
    trained_layer = trained.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        trained_layer.lora_A.fill_(0.123456789)
        trained_layer.lora_B.fill_(0.234567891)
    save_adapter(trained, str(tmp_path), _config(), adapter_format="peft")
    restored = peft.PeftModel.from_pretrained(base, tmp_path)
    layer = restored.base_model.model.model.layers[0].self_attn.q_proj
    assert layer.lora_A["default"].weight.dtype == torch.float32
    assert torch.equal(layer.lora_A["default"].weight, trained_layer.lora_A)
    assert torch.equal(layer.lora_B["default"].weight, trained_layer.lora_B)
    trained.eval()
    restored.eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(restored(inputs).logits, trained(inputs).logits)
