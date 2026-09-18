import torch

from nanoft import detect_device, get_recommended_dtype


def test_cuda_dtype_prefers_bfloat16_only_when_supported():
    assert get_recommended_dtype("cuda", supports_bf16=True) is torch.bfloat16
    assert get_recommended_dtype("cuda", supports_bf16=False) is torch.float16


def test_non_cuda_recommended_dtypes():
    assert get_recommended_dtype("mps") is torch.float32
    assert get_recommended_dtype("cpu") is torch.float32


def test_detect_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    info = detect_device()

    assert info.device_type == "mps"
    assert info.device_name == "Apple Metal"
    assert info.supports_fp16 is True
    assert info.supports_bf16 is False


def test_detect_device_prefers_cuda_over_mps(monkeypatch):
    class Properties:
        total_memory = 8 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: Properties(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index: (8, 0),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    info = detect_device()

    assert info.device_type == "cuda"
    assert info.device_name == "Test GPU"
    assert info.total_memory_gb == 8.0
    assert info.supports_bf16 is True
