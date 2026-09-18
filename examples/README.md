# Examples

`lora_sft_demo.py` 是 NanoFT 唯一的端到端 LoRA 示例入口。NanoFT 负责
LoRA 注入、adapter I/O 和合并导出；训练循环仍由 TRL `SFTTrainer` 执行。

运行示例前安装训练扩展：

```bash
pip install -e '.[trl]'
```

Qwen3-0.6B 与 Unified CHIP2 只是脚本的开箱即用默认值，不属于脚本名称或
实现约束。更换模型时，通过 `--model-name` 和 `--target-modules` 显式指定
模型及其目标 Linear 层。

```bash
# 训练，并保存 NanoFT 原生与 PEFT 兼容 adapter
python examples/lora_sft_demo.py train

# 对比 base、adapter 与 merged model
python examples/lora_sft_demo.py evaluate --variants base,native

# 合并 adapter，导出并重新加载校验标准 Transformers 模型
python examples/lora_sft_demo.py merge
```

CUDA 会在 `--dtype auto` 下选择 BF16/FP16，MPS 默认使用稳定的 FP32。
Apple Silicon 内存受限时可以显式传 `--dtype float16`，并先用 10-20 step
确认 loss 与梯度没有出现 NaN/Inf。

默认产物位于 `outputs/lora_sft_demo/`：

```text
outputs/lora_sft_demo/
├── trainer/
├── adapter_native/
├── adapter_peft/
├── merged_model/          # 使用 train --save-merged 或 merge 生成
├── run_metrics.json
└── evaluation_results.json
```

使用其他模型的示例：

```bash
python examples/lora_sft_demo.py train \
  --model-name /path/to/model \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --data-path /path/to/train.jsonl \
  --text-column text
```

完整说明见
[`docs/guides/lora-sft.md`](../docs/guides/lora-sft.md)。
