import torch
import torch.nn as nn
import pytest

from nanoft import (
    LoRAConfig,
    apply_lora,
    load_adapter,
    merge_and_unload_lora,
    save_adapter,
    save_peft_adapter,
    save_merged_model,
)
from nanoft.layers import LoRALinear
from safetensors.torch import load_file


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 3)
        self.out = nn.Linear(3, 2)

    def forward(self, x):
        return self.out(self.q_proj(x))


class SavePretrainedModel(TinyModel):
    def save_pretrained(self, save_dir):
        torch.save(self.state_dict(), f"{save_dir}/model.pt")


def _fill_lora(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.fill_(0.25)
                module.lora_B.fill_(0.5)


def test_merge_and_unload_replaces_lora_layers_and_preserves_output():
    torch.manual_seed(0)
    model = apply_lora(TinyModel(), LoRAConfig(r=2, lora_alpha=4))
    _fill_lora(model)
    x = torch.randn(5, 4)

    before = model(x)
    merged = merge_and_unload_lora(model, inplace=False)
    after = merged(x)

    assert torch.allclose(before, after, atol=1e-6)
    assert not any(isinstance(module, LoRALinear) for module in merged.modules())
    assert not any("lora_A" in key or "lora_B" in key for key in merged.state_dict())


def test_save_merged_model_saves_plain_state_dict(tmp_path):
    model = apply_lora(SavePretrainedModel(), LoRAConfig(r=2, lora_alpha=4))
    _fill_lora(model)

    save_merged_model(model, str(tmp_path), inplace=True)
    saved = torch.load(tmp_path / "model.pt", weights_only=True)

    assert saved
    assert not any("lora_A" in key or "lora_B" in key for key in saved)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())


def test_load_adapter_injects_lora_when_needed(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    trained = apply_lora(TinyModel(), config)
    _fill_lora(trained)
    save_adapter(trained, str(tmp_path), config)

    loaded = load_adapter(TinyModel(), str(tmp_path))

    assert isinstance(loaded.q_proj, LoRALinear)
    assert torch.allclose(loaded.q_proj.lora_A, trained.q_proj.lora_A)
    assert torch.allclose(loaded.q_proj.lora_B, trained.q_proj.lora_B)


def test_save_adapter_defaults_to_native_keys(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    trained = apply_lora(TinyModel(), config)

    save_adapter(trained, str(tmp_path), config)
    saved = load_file(tmp_path / "adapter_model.safetensors")

    assert "q_proj.lora_A" in saved
    assert not any(key.startswith("base_model.model.") for key in saved)


def test_save_peft_adapter_uses_peft_keys_and_loads_back(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    trained = apply_lora(TinyModel(), config)
    _fill_lora(trained)

    save_peft_adapter(trained, str(tmp_path), config)
    saved = load_file(tmp_path / "adapter_model.safetensors")
    loaded = load_adapter(TinyModel(), str(tmp_path))

    assert "base_model.model.q_proj.lora_A.default.weight" in saved
    assert isinstance(loaded.q_proj, LoRALinear)
    assert torch.allclose(loaded.q_proj.lora_A, trained.q_proj.lora_A)
    assert torch.allclose(loaded.q_proj.lora_B, trained.q_proj.lora_B)


def test_load_adapter_raises_on_incompatible_target(tmp_path):
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    trained = apply_lora(TinyModel(), config)
    save_adapter(trained, str(tmp_path), config)

    incompatible = apply_lora(TinyModel(), LoRAConfig(r=2, target_modules=["out"]))

    with pytest.raises(RuntimeError, match="Adapter weights do not match model"):
        load_adapter(incompatible, str(tmp_path))
