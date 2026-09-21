import copy

import torch
import pytest
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from nanoft import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    load_adapter,
    save_adapter,
    save_merged_model,
    save_peft_adapter,
)


peft = pytest.importorskip("peft")


def _tiny_llama(*, with_bias=False):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_bias=with_bias,
        mlp_bias=with_bias,
    )
    return LlamaForCausalLM(config)


def test_nanoft_peft_export_loads_with_peft(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    nanoft_model = apply_lora(_tiny_llama(), config)
    nanoft_layer = nanoft_model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        nanoft_layer.lora_A.fill_(0.25)
        nanoft_layer.lora_B.fill_(0.5)

    save_peft_adapter(nanoft_model, str(tmp_path), config)

    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        in saved
    )
    peft_model = peft.PeftModel.from_pretrained(_tiny_llama(), tmp_path)
    peft_layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj

    assert torch.allclose(
        peft_layer.lora_A["default"].weight,
        nanoft_layer.lora_A,
    )
    assert torch.allclose(
        peft_layer.lora_B["default"].weight,
        nanoft_layer.lora_B,
    )


def test_peft_export_targets_only_injected_modules(tmp_path):
    class VisionProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, hidden_states):
            return self.linear(hidden_states)

    base_model = _tiny_llama()
    base_model.vision = torch.nn.Module()
    base_model.vision.q_proj = VisionProjection()
    restored_base = copy.deepcopy(base_model)
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["v_proj", "q_proj"])
    original_config = copy.deepcopy(config)

    with pytest.raises(ValueError, match="is not supported"):
        peft.get_peft_model(
            copy.deepcopy(base_model),
            peft.LoraConfig(r=2, lora_alpha=4, target_modules=config.target_modules),
        )

    trained = apply_lora(base_model, config)
    expected_targets = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    with torch.no_grad():
        for name in expected_targets:
            layer = trained.get_submodule(name)
            layer.lora_A.fill_(0.25)
            layer.lora_B.fill_(0.5)
    trained.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected_logits = trained(input_ids).logits

    save_peft_adapter(trained, str(tmp_path / "peft"), config)
    save_adapter(trained, str(tmp_path / "native"), config)
    assert config == original_config
    peft_config = LoRAConfig.load(tmp_path / "peft" / "adapter_config.json")
    native_config = LoRAConfig.load(tmp_path / "native" / "adapter_config.json")
    assert peft_config.target_modules == expected_targets
    assert native_config.target_modules == original_config.target_modules

    restored = peft.PeftModel.from_pretrained(restored_base, tmp_path / "peft")
    restored.eval()
    restored_targets = {
        name for name, module in restored.base_model.model.named_modules()
        if hasattr(module, "lora_A")
    }
    assert restored_targets == set(expected_targets)
    for name in expected_targets:
        actual_layer = restored.base_model.model.get_submodule(name)
        expected_layer = trained.get_submodule(name)
        assert torch.equal(
            actual_layer.lora_A["default"].weight, expected_layer.lora_A
        )
        assert torch.equal(
            actual_layer.lora_B["default"].weight, expected_layer.lora_B
        )
    with torch.no_grad():
        assert torch.allclose(restored(input_ids).logits, expected_logits, atol=1e-6)


def test_peft_export_loads_with_nanoft(tmp_path):
    peft_config = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj"],
    )
    peft_model = peft.get_peft_model(_tiny_llama(), peft_config)
    peft_layer = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        peft_layer.lora_A["default"].weight.fill_(0.25)
        peft_layer.lora_B["default"].weight.fill_(0.5)
    peft_model.save_pretrained(tmp_path, save_embedding_layers=False)

    nanoft_model = load_adapter(_tiny_llama(), str(tmp_path))
    nanoft_layer = nanoft_model.model.layers[0].self_attn.q_proj

    assert torch.allclose(
        nanoft_layer.lora_A,
        peft_layer.lora_A["default"].weight,
    )
    assert torch.allclose(
        nanoft_layer.lora_B,
        peft_layer.lora_B["default"].weight,
    )


def test_nanoft_bias_all_export_loads_with_peft(tmp_path):
    config = LoRAConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj"],
        bias="all",
    )
    nanoft_model = apply_lora(_tiny_llama(with_bias=True), config)
    with torch.no_grad():
        nanoft_model.model.layers[0].self_attn.q_proj.bias.fill_(0.25)
        nanoft_model.model.layers[0].self_attn.k_proj.bias.fill_(0.5)

    save_peft_adapter(nanoft_model, str(tmp_path), config)
    peft_model = peft.PeftModel.from_pretrained(
        _tiny_llama(with_bias=True),
        tmp_path,
    )
    attention = peft_model.base_model.model.model.layers[0].self_attn

    assert torch.allclose(
        attention.q_proj.base_layer.bias,
        nanoft_model.model.layers[0].self_attn.q_proj.bias,
    )
    assert torch.allclose(
        attention.k_proj.bias,
        nanoft_model.model.layers[0].self_attn.k_proj.bias,
    )


def test_peft_bias_all_export_loads_with_nanoft(tmp_path):
    peft_config = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj"],
        bias="all",
    )
    peft_model = peft.get_peft_model(
        _tiny_llama(with_bias=True),
        peft_config,
    )
    peft_attention = peft_model.base_model.model.model.layers[0].self_attn
    with torch.no_grad():
        peft_attention.q_proj.base_layer.bias.fill_(0.25)
        peft_attention.k_proj.bias.fill_(0.5)
    peft_model.save_pretrained(tmp_path, save_embedding_layers=False)

    nanoft_model = load_adapter(
        _tiny_llama(with_bias=True),
        str(tmp_path),
    )
    nanoft_attention = nanoft_model.model.layers[0].self_attn

    assert torch.allclose(
        nanoft_attention.q_proj.bias,
        peft_attention.q_proj.base_layer.bias,
    )
    assert torch.allclose(
        nanoft_attention.k_proj.bias,
        peft_attention.k_proj.bias,
    )


def test_native_adapter_preserves_tiny_llama_output(tmp_path):
    base_model = _tiny_llama()
    base_state = {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    trained = apply_lora(base_model, config)
    with torch.no_grad():
        trained.model.layers[0].self_attn.q_proj.lora_A.fill_(0.25)
        trained.model.layers[0].self_attn.q_proj.lora_B.fill_(0.5)
    trained.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = trained(input_ids).logits

    save_adapter(trained, str(tmp_path), config)
    restored_base = _tiny_llama()
    restored_base.load_state_dict(base_state)
    restored = load_adapter(restored_base, str(tmp_path))
    restored.eval()
    actual = restored(input_ids).logits

    assert torch.allclose(actual, expected, atol=1e-6)


def test_merged_tiny_llama_round_trip_is_non_destructive_by_default(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    model = apply_lora(_tiny_llama(), config)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.lora_A.fill_(0.25)
        model.model.layers[0].self_attn.q_proj.lora_B.fill_(0.5)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids).logits

    save_merged_model(model, str(tmp_path))
    restored = LlamaForCausalLM.from_pretrained(tmp_path)
    restored.eval()
    actual = restored(input_ids).logits

    assert isinstance(model.model.layers[0].self_attn.q_proj, LoRALinear)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_inplace_merged_model_matches_reloaded_model(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    model = apply_lora(_tiny_llama(), config)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.lora_A.fill_(0.25)
        model.model.layers[0].self_attn.q_proj.lora_B.fill_(0.5)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    save_merged_model(model, str(tmp_path), inplace=True)
    in_memory_merged_logits = model(input_ids).logits
    restored = LlamaForCausalLM.from_pretrained(tmp_path)
    restored.eval()
    reloaded_logits = restored(input_ids).logits

    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert torch.allclose(
        reloaded_logits,
        in_memory_merged_logits,
        atol=1e-6,
    )
