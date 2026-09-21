# 模型准备与 LoRA 层

## prepare_model_for_training

```python
model = prepare_model_for_training(model, config)
```

该函数：

1. 冻结模型现有参数。
2. 根据 `config.target_modules` 替换普通 `nn.Linear`。
3. 开放 `lora_A`、`lora_B` 和配置指定的 bias。

模型会原地修改，同时返回同一个模型对象以便链式使用。

## apply_lora

```python
model = apply_lora(model, config)
```

当前实现同样会执行 base 参数冻结和 adapter 参数开放。它是更底层的注入
入口；常规训练代码优先使用 `prepare_model_for_training()`，以表达调用
意图。

可以通过 `target_modules` 参数临时覆盖配置中的目标模块：

```python
apply_lora(model, config, target_modules=["q_proj", "v_proj"])
```

## LoRALinear

`LoRALinear` 继承 `torch.nn.Linear`，保留 base `weight` 和可选 `bias`，
新增两个直接参数：

```text
lora_A
lora_B
```

NanoFT 原生 key 不包含 adapter runtime 名称，也不包含额外 `.weight`
后缀。PEFT 兼容转换由保存/加载函数负责。

初始状态使用 Kaiming 初始化 `lora_A`，将 `lora_B` 初始化为零，因此刚
注入时 LoRA 分支不会改变 base model 输出。

## remove_lora

```python
model = remove_lora(model)
```

该函数会先合并尚未合并的 LoRA delta，再用普通 `nn.Linear` 替换
`LoRALinear`。它不是“丢弃 adapter”；调用后保留的是合并结果。

## 约束

- 不支持重复注入。
- 只匹配普通 `nn.Linear`。
- `fan_in_fan_out=True` 当前会在配置验证阶段报错。
- `eval()` 不会自动合并由 `apply_lora()` 创建的层，因为这些层使用
  `merge_weights=False`。需要显式调用 merge API。
