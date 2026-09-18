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

## 安全检查

NanoFT 会拒绝：

- 非 tensor adapter 值。
- 包含 NaN 或 Inf 的浮点/复数 tensor。
- 与模型期望 shape 不一致的 tensor。
- 严格模式下缺失或额外的 adapter key。
