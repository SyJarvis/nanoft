# Adapter I/O

## save_adapter

```python
save_adapter(
    model,
    "outputs/adapter_native",
    config,
    safe_serialization=True,
)
```

默认写出：

```text
adapter_native/
├── adapter_config.json
└── adapter_model.safetensors
```

如果环境没有 safetensors，`safe_serialization=True` 会回退到
`adapter_model.bin`。

`adapter_format` 可选 `"nanoft"` 或 `"peft"`，但调用方通常应使用语义
更清楚的 `save_peft_adapter()`。

## save_peft_adapter

```python
save_peft_adapter(model, "outputs/adapter_peft", config)
```

该函数写出 PEFT 兼容 state dict key，可由 `PeftModel.from_pretrained()`
加载。

## load_adapter

```python
model = load_adapter(
    base_model,
    "outputs/adapter_native",
    strict=True,
)
```

加载顺序：

1. 读取 `adapter_config.json`。
2. 如果模型还没有 LoRA 层，根据配置进行注入。
3. 读取 safetensors 或 PyTorch bin。
4. 将 PEFT key 转换为 NanoFT 内部 key。
5. 校验 key、shape、tensor 类型和有限值。
6. 将 adapter 权重载入模型。

建议保持 `strict=True`。只有明确需要诊断部分 adapter 时才关闭严格模式。

权重文件保留保存时的 tensor dtype。若 adapter 为 FP32、基座为 BF16，
请在加载时显式指定 dtype，避免将 FP32 权重写入 BF16 adapter 参数：

```python
import torch

model = load_adapter(
    base_model,
    "outputs/adapter_native",
    adapter_dtype=torch.float32,
)
```

同样适用于 PEFT 格式权重及已经注入 dense LoRA 的模型。该选项仅转换
LoRA A/B，不转换基座或 bias；继续训练时应在加载后构造 optimizer。
`adapter_dtype` 是运行时选项，不写入 `adapter_config.json`，因此不会向
PEFT 配置添加专有字段。

默认保持原有行为：新注入的 adapter 跟随基座 dtype，已注入的 adapter
保留其现有 dtype。`adapter_dtype` 仅支持 dense LoRA，量化 adapter
传入该选项会被拒绝。

## 安全检查

NanoFT 会拒绝：

- 非 tensor adapter 值。
- 包含 NaN 或 Inf 的浮点/复数 tensor。
- 与模型期望 shape 不一致的 tensor。
- 严格模式下缺失或额外的 adapter key。
