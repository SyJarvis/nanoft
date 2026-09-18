# 使用 TRL 进行 LoRA SFT

NanoFT 只保留一个端到端示例：`examples/lora_sft_demo.py`。它使用三个
子命令覆盖训练、评估和合并导出，避免为每个模型复制一组脚本。

Qwen3-0.6B 是默认演示模型，Unified CHIP2 是默认演示数据。两者都可以通过
命令行替换。

## 安装

先安装 NanoFT，再安装训练命令使用的外部框架：

```bash
pip install -e '.[trl]'
```

NanoFT 不在核心依赖中锁定 TRL 或 Accelerate 的具体版本。请为当前
Transformers/PyTorch 环境选择彼此兼容的版本。`evaluate` 和 `merge`
子命令不依赖 TRL。

## 训练

先用 10 至 20 step 验证数据、显存和保存链路：

```bash
python examples/lora_sft_demo.py train \
  --max-seq-length 512 \
  --max-samples 128 \
  --max-steps 20
```

默认配置使用 2048 个样本、batch size 1、gradient accumulation 4，因此
有效 batch 为 4，一轮约 512 step。链路稳定后，建议先测试
`256-1024` step（约 0.5-2 epoch），最终根据 held-out loss 与生成质量选择
checkpoint，而不是固定追求某个 step。

常规运行：

```bash
python examples/lora_sft_demo.py train --max-steps 512
```

训练完成后会保存：

- `adapter_native/`：NanoFT 原生 adapter。
- `adapter_peft/`：PEFT 兼容 adapter。
- `run_metrics.json`：运行参数和训练指标。
- `trainer/`：TRL 管理的 checkpoint。

如希望训练结束时同时写出完整模型，可添加 `--save-merged`。更推荐先保存
adapter，评估确认后再独立执行 `merge`。

## 使用其他模型或数据

```bash
python examples/lora_sft_demo.py train \
  --model-name /path/to/model \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --data-path /path/to/train.jsonl \
  --text-column text \
  --output-root outputs/my_run
```

数据列应当包含已经格式化好的完整训练文本。不同模型架构的 Linear 层名称
不同，因此使用非默认模型时应检查模型结构并显式调整 `--target-modules`。

## 评估

默认比较 base 与 NanoFT native adapter：

```bash
python examples/lora_sft_demo.py evaluate --variants base,native
```

如果已经生成所有格式：

```bash
python examples/lora_sft_demo.py evaluate \
  --variants base,native,peft,merged
```

评估结果包括 held-out token loss、perplexity 和确定性生成，并写入
`evaluation_results.json`。默认从训练样本之后的第 2048 行开始取评估集，
可用 `--eval-offset` 和 `--eval-samples` 修改。

默认数据使用 `<human>/<bot>` prompt。对于其他数据格式，可用
`--prompt-template` 传入包含 `{prompt}` 的模板，或通过 `--prompts-file`
提供 JSONL 测试问题。

## 合并并导出

```bash
python examples/lora_sft_demo.py merge
```

该命令加载 base model 与 NanoFT/PEFT adapter，将 LoRA 分支合并到普通
Linear 权重，写出标准 Transformers 模型，并重新加载产物校验 logits。

指定其他运行目录时：

```bash
python examples/lora_sft_demo.py merge \
  --model-name /path/to/model \
  --adapter outputs/my_run/adapter_native \
  --output-dir outputs/my_run/merged_model
```

脚本会分别报告：

- adapter model 到内存中 merged model 的低精度 merge 漂移，仅作诊断；
- 内存中 merged model 到重新加载产物的一致性，决定序列化校验是否通过。
