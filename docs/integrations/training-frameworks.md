# TRL 与训练框架

NanoFT 准备标准 PyTorch/Transformers 模型，外部框架直接接收该模型。项目
不会提供自己的 Trainer wrapper。

## TRL

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs/trainer",
        dataset_text_field="text",
        max_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=100,
    ),
)
trainer.train()
```

这里的 `model` 已由 NanoFT 的 `prepare_model_for_training()` 注入 LoRA。
TRL 负责反向传播、optimizer、checkpoint 和日志。训练后仍使用 NanoFT
保存 adapter。

完整可执行代码见仓库中的 `examples/lora_sft_demo.py train`。

## Transformers Trainer

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="outputs/trainer"),
    train_dataset=tokenized_dataset,
)
trainer.train()
```

NanoFT 不依赖特定 Trainer；只要外部训练循环优化
`requires_grad=True` 的参数即可。

## 纯 PyTorch

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=2e-4)
```

训练循环完全属于调用方。NanoFT 不提供 optimizer、scheduler 或
gradient accumulation 抽象。

## 版本策略

NanoFT 核心依赖 Transformers，但不把 TRL、Accelerate 或 PEFT 固定到
某个精确版本。外部训练框架变化较快，应在具体训练项目中维护一组经过验证
的 lock file；NanoFT 示例只使用公开集成表面。
