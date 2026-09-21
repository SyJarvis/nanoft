import pytest
import torch
import torch.nn as nn

from nanoft import LoRAConfig, apply_lora, remove_lora
from nanoft.layers import LoRALinear


class MatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(4, 3)
        self.block.q_proj_extra = nn.Linear(4, 3)
        self.output = nn.Linear(3, 2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"r": 0}, "r must be a positive integer"),
        ({"lora_alpha": 0}, "lora_alpha must be greater than 0"),
        ({"lora_dropout": -0.1}, "lora_dropout must be in the range"),
        ({"lora_dropout": 1.0}, "lora_dropout must be in the range"),
        ({"target_modules": []}, "target_modules must be None"),
        ({"bias": "invalid"}, "bias must be"),
        ({"fan_in_fan_out": True}, "fan_in_fan_out is not supported"),
    ],
)
def test_lora_config_rejects_unsupported_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LoRAConfig(**kwargs)


def test_apply_lora_uses_module_suffix_matching():
    model = apply_lora(
        MatchingModel(),
        LoRAConfig(r=2, target_modules=["q_proj"]),
    )

    assert isinstance(model.block.q_proj, LoRALinear)
    assert isinstance(model.block.q_proj_extra, nn.Linear)
    assert not isinstance(model.block.q_proj_extra, LoRALinear)


def test_apply_lora_raises_when_no_modules_match_without_freezing_model():
    model = MatchingModel()

    with pytest.raises(ValueError, match="No nn.Linear modules matched"):
        apply_lora(
            model,
            LoRAConfig(r=2, target_modules=["missing"]),
        )

    assert all(param.requires_grad for param in model.parameters())


def test_apply_lora_rejects_repeated_injection_without_replacing_weights():
    model = apply_lora(
        MatchingModel(),
        LoRAConfig(r=2, target_modules=["q_proj"]),
    )
    original_lora_a = model.block.q_proj.lora_A

    with pytest.raises(ValueError, match="already contains LoRA layers"):
        apply_lora(
            model,
            LoRAConfig(r=2, target_modules=["q_proj"]),
        )

    assert model.block.q_proj.lora_A is original_lora_a


@pytest.mark.parametrize(
    ("bias", "expected_biases"),
    [
        ("none", set()),
        ("lora_only", {"block.q_proj.bias"}),
        (
            "all",
            {
                "block.q_proj.bias",
                "block.q_proj_extra.bias",
                "output.bias",
            },
        ),
    ],
)
def test_apply_lora_honors_bias_strategy(bias, expected_biases):
    model = apply_lora(
        MatchingModel(),
        LoRAConfig(r=2, target_modules=["q_proj"], bias=bias),
    )
    trainable_biases = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.endswith(".bias")
    }

    assert trainable_biases == expected_biases


def test_remove_lora_preserves_layer_state():
    model = MatchingModel().to(dtype=torch.float64)
    model.eval()
    model = apply_lora(
        model,
        LoRAConfig(r=2, target_modules=["q_proj"], bias="lora_only"),
    )
    lora_layer = model.block.q_proj

    model = remove_lora(model)
    restored = model.block.q_proj

    assert type(restored) is nn.Linear
    assert restored.weight.device == lora_layer.weight.device
    assert restored.weight.dtype == lora_layer.weight.dtype
    assert restored.weight.requires_grad == lora_layer.weight.requires_grad
    assert restored.bias.requires_grad == lora_layer.bias.requires_grad
    assert restored.training == lora_layer.training
