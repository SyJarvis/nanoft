# 适配其他模型

统一 Demo 默认使用 Qwen3-0.6B，但脚本和 NanoFT API 不绑定 Qwen。适配其他
模型时，主要需要确认模型加载方式、目标 Linear 名称和训练数据格式。

## 查找目标模块

先列出模型中的 Linear：

```python
import torch.nn as nn

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name, module.in_features, module.out_features)
```

`target_modules` 支持完整模块名或点号分隔名称的后缀匹配。例如
`"q_proj"` 可以匹配 `model.layers.0.self_attn.q_proj`，但不会匹配
`some_q_proj_extra`。

推荐从 attention projection 开始：

```python
config = LoRAConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

如果模型的 MLP 也使用普通 `nn.Linear`，可以加入 `gate_proj`、`up_proj`
和 `down_proj`。模块名由模型实现决定，不应直接假设所有架构相同。

## 使用统一 Demo

```bash
python examples/lora_sft_demo.py train \
  --model-name /path/to/model \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --data-path /path/to/train.jsonl \
  --text-column text \
  --output-root outputs/my_model
```

数据的 `text` 列应当已经是模型需要看到的完整序列。NanoFT 当前不会自动
选择 chat template；请在数据准备阶段调用 tokenizer 的
`apply_chat_template()` 或提前生成文本。

## 常见失败

### 没有匹配到模块

```text
No nn.Linear modules matched target_modules
```

检查模型是否使用其他 Linear 类型、量化层或不同命名。NanoFT 0.2.0 只会
替换普通 `torch.nn.Linear`。

### 重复注入

同一个模型不能连续调用两次 `apply_lora()` 或
`prepare_model_for_training()`。加载 adapter 时，如果模型已经包含
`LoRALinear`，NanoFT 会直接加载权重，不会再次注入。

### Base model 不一致

Adapter 的 shape 和模块路径必须与 base model 一致。建议在
`LoRAConfig.base_model_name_or_path` 中记录模型标识，并固定模型 revision。
