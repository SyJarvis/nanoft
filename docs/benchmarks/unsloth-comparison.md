# Unsloth 对比记录

{% hint style="warning" %}
这是一次特定 Mac/MPS 环境的历史运行记录，不代表跨硬件的通用性能结论。
{% endhint %}

## 对齐配置

- Model: `Qwen/Qwen2.5-0.5B`
- Dataset: `data/unified_chip2.jsonl`
- Samples: 2048
- Max length: 512
- Steps: 100
- Batch size: 1
- Gradient accumulation steps: 4
- Precision: fp32
- LoRA rank: 8
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## NanoFT 结果

- Status: completed
- Final train loss: 1.5689
- Final eval loss: 1.4979
- Final grad norm: 0.6758
- Adapter size: 8.42 MB
- Merged model size: 942.32 MB
- Adapter tensors: finite
- Merged tensors: finite
- Merged LoRA keys: 0

## 此 Mac 上的 Unsloth 结果

- Status: unsupported for this TRL training path
- Installed packages:
  - `unsloth 2026.6.9`
  - `trl 0.24.0`
  - `bitsandbytes 0.49.2`
  - `torch 2.7.1`
- Unsloth successfully loaded the model through `mlx-lm`.
- Unsloth successfully applied LoRA with 4,399,104 trainable parameters.
- The returned model is an MLX-backed object without HuggingFace `model.config`.
- `trl.SFTTrainer` expects a HuggingFace `PreTrainedModel`, so training cannot proceed through the demo-style TRL path.

Recorded error:

```text
FastLanguageModel returned a non-HuggingFace model without `config`;
TRL SFTTrainer cannot train this MLX-backed model.
```

## 当前结论

On this Mac/MPS setup, NanoFT completes the matched 100-step run and produces a normal merged HuggingFace model. Unsloth can initialize via MLX, but the CUDA/TRL-style training script is not directly compatible with the MLX-backed model returned in this environment.

For a throughput and memory-speed comparison, run both frameworks on a CUDA machine where Unsloth returns a HuggingFace-compatible model for TRL training. For a Mac-local comparison, the next step is to find or build an MLX-native Unsloth training loop and compare that against NanoFT.
