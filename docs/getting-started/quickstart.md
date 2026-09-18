# Quick Start

下面的最小流程完成模型加载、LoRA 注入、外部训练以及 adapter 保存。示例
模型可以替换，关键是 `target_modules` 必须匹配实际模型中的
`nn.Linear` 名称。

## 1. 加载模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## 2. 注入 LoRA

```python
from nanoft import (
    LoRAConfig,
    prepare_model_for_training,
    print_trainable_parameters,
)

config = LoRAConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    base_model_name_or_path=model_name,
)

model = prepare_model_for_training(model, config)
print_trainable_parameters(model)
```

`prepare_model_for_training()` 会原地替换目标 Linear、冻结 base 参数，并
开放 LoRA 参数。若没有模块匹配，NanoFT 会直接报错。

## 3. 交给训练框架

```python
# trainer = SFTTrainer(model=model, ...)
# trainer.train()
```

训练循环、optimizer、gradient accumulation、checkpoint 和日志都由外部
框架负责。完整 TRL 示例见[使用 TRL 进行 LoRA SFT](../guides/lora-sft.md)。

## 4. 保存结果

```python
from nanoft import save_adapter, save_peft_adapter, save_merged_model

save_adapter(model, "outputs/adapter_native", config)
save_peft_adapter(model, "outputs/adapter_peft", config)
save_merged_model(
    model,
    "outputs/merged_model",
    tokenizer=tokenizer,
)
```

- `adapter_native` 面向 NanoFT 的直接加载。
- `adapter_peft` 使用标准 PEFT key，可交给 `PeftModel`。
- `merged_model` 是不包含 LoRA 层的标准完整模型。

默认合并导出会复制模型以保留当前 adapter model。大型模型内存不足时，
可以传 `inplace=True`，但原模型会被不可逆地合并并卸载。
