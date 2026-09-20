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

`split_sft_data.py` 是训练前的数据准备工具，不执行训练。它读取已经按模型
格式化的 `prompt` / `completion` 字符串，按原始完整 prompt 分组，避免同一
prompt 的不同答案跨越训练/验证边界：

```bash
python examples/split_sft_data.py \
  --input data/formatted.jsonl \
  --output-dir data/grouped \
  --validation-size 2048 --seed 3407
```

输出目录必须不存在，结果为 `train.jsonl`、`validation.jsonl`、`manifest.json`。
整组保留可能使验证行数超过目标值；不删除重复记录或规范化 prompt，也不保证
没有语义重合。原始 messages、图像和音频不属于此入口支持的数据格式。

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

对于上述分组文件，将数据参数改为：

```bash
python examples/lora_sft_demo.py train \
  --model-name /path/to/model \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --data-path data/grouped/train.jsonl \
  --eval-data-path data/grouped/validation.jsonl \
  --data-format prompt-completion --completion-only-loss \
  --dtype bfloat16 --adapter-dtype float32 \
  --amp auto --full-determinism --attn-implementation eager
```

训练、独立评估和合并应保持相同的精度与执行选项；评估独立验证文件时设置
`--eval-offset 0`，同时使用目标模型的生成模板。`--adapter-dtype` 控制 A/B
参数精度，`--amp` 控制 autocast；CLI 在 `auto` 加载模式下保留 checkpoint
的唯一 LoRA dtype。效果是否改善需由任务评估判断，不能仅根据 loss 下降。

完整说明见
[`docs/guides/lora-sft.md`](../docs/guides/lora-sft.md)。
