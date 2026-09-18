import pytest
import torch
import torch.nn.functional as F

from nanoft import LoRALinear


def _filled_layer(*, dropout=0.0, merge_weights=False, dtype=torch.float32):
    layer = LoRALinear(
        4,
        3,
        r=2,
        lora_alpha=4,
        lora_dropout=dropout,
        merge_weights=merge_weights,
        dtype=dtype,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(12, dtype=dtype).reshape(3, 4) / 10
        )
        layer.bias.copy_(torch.tensor([0.1, 0.2, 0.3], dtype=dtype))
        layer.lora_A.fill_(0.25)
        layer.lora_B.fill_(0.5)
    return layer


def test_lora_forward_matches_formula():
    layer = _filled_layer()
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10

    expected = F.linear(inputs, layer.weight, layer.bias)
    expected += (
        inputs @ layer.lora_A.T @ layer.lora_B.T
    ) * layer.scaling

    assert torch.allclose(layer(inputs), expected)


def test_lora_backward_only_trains_adapter_parameters():
    layer = _filled_layer()
    inputs = torch.randn(2, 4)

    layer(inputs).sum().backward()

    assert layer.weight.grad is None
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert torch.isfinite(layer.lora_A.grad).all()
    assert torch.isfinite(layer.lora_B.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_merge_unmerge_is_numerically_stable(dtype):
    layer = _filled_layer(dtype=dtype)
    base_weight = layer.weight.detach().clone()
    inputs = torch.randn(2, 4, dtype=dtype)
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    before = layer(inputs)

    layer.merge()
    merged = layer(inputs)
    layer.unmerge()
    after = layer(inputs)

    assert torch.allclose(before, merged, atol=tolerance, rtol=tolerance)
    assert torch.allclose(before, after, atol=tolerance, rtol=tolerance)
    assert torch.allclose(layer.weight, base_weight, atol=tolerance, rtol=tolerance)


def test_train_eval_return_self_and_manage_merge_state():
    layer = _filled_layer(merge_weights=True)

    assert layer.eval() is layer
    assert layer.merged is True
    assert layer.train() is layer
    assert layer.merged is False


def test_dropout_is_active_only_during_training():
    layer = _filled_layer(dropout=0.5)
    inputs = torch.ones(32, 4)

    layer.train()
    torch.manual_seed(0)
    first = layer(inputs)
    torch.manual_seed(1)
    second = layer(inputs)
    layer.eval()
    eval_first = layer(inputs)
    eval_second = layer(inputs)

    assert not torch.allclose(first, second)
    assert torch.allclose(eval_first, eval_second)
