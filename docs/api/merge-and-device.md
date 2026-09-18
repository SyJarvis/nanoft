# 合并与设备工具

## merge_lora_weights

```python
merged = merge_lora_weights(model, inplace=False)
```

将 LoRA delta 折叠进 base 权重，但层类型仍然是 `LoRALinear`，adapter 参数
也仍然存在，因此可以调用 `unmerge_lora_weights()` 反向恢复。

## merge_and_unload_lora

```python
standard_model = merge_and_unload_lora(model, inplace=False)
```

合并权重后用普通 `nn.Linear` 替换所有 LoRA 层。返回模型不再包含
`lora_A` 或 `lora_B`，适合标准模型导出。

## save_merged_model

```python
save_merged_model(
    model,
    "outputs/merged_model",
    tokenizer=tokenizer,
    inplace=False,
)
```

如果模型支持 `save_pretrained()`，使用 Transformers 格式保存；否则写出
PyTorch state dict。传入 tokenizer 时会一起保存。

`inplace=False` 会深拷贝模型，安全但峰值内存较高。`inplace=True` 适合
一次性导出，调用后原对象已经失去 LoRA 层。

## 设备 API

```python
from nanoft import detect_device, get_recommended_dtype, move_to_device

info = detect_device()
dtype = get_recommended_dtype(
    info.device_type,
    supports_bf16=info.supports_bf16,
)
model = move_to_device(model, info.device_type, dtype)
```

`detect_device()` 的优先级是 CUDA、MPS、CPU，返回 `DeviceInfo`：

| 字段 | 说明 |
|---|---|
| `device_type` | `cuda`、`mps` 或 `cpu` |
| `device_name` | 设备名称 |
| `total_memory_gb` | 可查询时的总显存 |
| `supports_fp16` | FP16 能力提示 |
| `supports_bf16` | BF16 能力提示 |

这些值是便捷默认，不替代训练框架的精度和 device map 配置。

`get_recommended_dtype()` 为 CUDA 选择 BF16/FP16，为 MPS 和 CPU 返回
稳定默认 FP32。MPS 用户可以在经过短训练验证后显式使用 FP16，详情见
[CUDA 与 MPS 支持](../getting-started/device-support.md)。
