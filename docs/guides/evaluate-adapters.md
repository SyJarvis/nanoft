# 加载与评估 Adapter

训练完成后应至少检查 held-out loss、确定性生成，以及 merge 后模型能否
重新加载。仓库统一 Demo 已实现这条验证路径。

## 使用统一 Demo

```bash
python examples/lora_sft_demo.py evaluate \
  --variants base,native
```

可用 variant：

- `base`：原始模型。
- `native`：NanoFT 原生 adapter。
- `peft`：由 PEFT `PeftModel` 加载的兼容 adapter。
- `merged`：合并导出的标准模型。

完整比较：

```bash
python examples/lora_sft_demo.py evaluate \
  --variants base,native,peft,merged \
  --output-root outputs/lora_sft_demo
```

## 在 Python 中加载

```python
from nanoft import load_adapter
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = load_adapter(base, "outputs/lora_sft_demo/adapter_native")
model.eval()
```

`load_adapter()` 会读取 `adapter_config.json`，在需要时注入 LoRA，再加载
NanoFT 原生或 PEFT 兼容 key。默认 `strict=True`，缺失、额外或 shape
不匹配的 adapter tensor 会触发错误。

## 如何选择训练 step

先计算：

```text
effective_batch = batch_size × gradient_accumulation × world_size
steps_per_epoch = ceil(train_samples / effective_batch)
```

建议：

- 10-20 step：只验证训练与保存链路。
- 约 0.5 epoch：第一次有意义的质量检查。
- 1-2 epoch：常见起始搜索范围。
- 最终 checkpoint：根据 held-out loss 和固定 prompts 的生成质量选择。

如果训练 loss 下降但 held-out loss 上升，应优先考虑过拟合，而不是继续增加
step。
