# LoRAConfig

`LoRAConfig` 是 NanoFT LoRA 行为和 adapter 序列化的单一配置来源。

```python
from nanoft import LoRAConfig

config = LoRAConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    base_model_name_or_path="Qwen/Qwen3-0.6B",
)
```

## 字段

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `r` | `8` | 低秩维度，必须是正整数 |
| `lora_alpha` | `16` | LoRA 缩放参数，必须大于 0 |
| `lora_dropout` | `0.0` | LoRA 分支 dropout，范围 `[0, 1)` |
| `target_modules` | `None` | 完整名称或点号后缀；默认 q/k/v projection |
| `fan_in_fan_out` | `False` | 当前必须保持 `False` |
| `bias` | `"none"` | `"none"`、`"all"` 或 `"lora_only"` |
| `task_type` | `"CAUSAL_LM"` | 写入 adapter 配置的任务类型 |
| `base_model_name_or_path` | `None` | 用于记录 base model |
| `inference_mode` | `False` | 保存 adapter 时副本会设置为 `True` |

实际 LoRA scaling 为：

```text
lora_alpha / r
```

## Bias 策略

- `none`：只有 `lora_A` 与 `lora_B` 可训练和保存。
- `lora_only`：额外训练并保存被 LoRA 替换层的 bias。
- `all`：训练并保存模型中所有名称以 `.bias` 结尾的参数。

## 序列化

```python
config.save("adapter_config.json")
restored = LoRAConfig.load("adapter_config.json")
```

`to_peft_dict()` 会加入 PEFT adapter 配置需要的兼容字段。
