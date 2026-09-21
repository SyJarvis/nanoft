# CUDA 与 MPS 支持

NanoFT 面向端侧大模型微调，正式支持两类 GPU 后端：

- **CUDA**：包括常规 NVIDIA GPU 和 Jetson。作为主要训练、验证和后续
  QLoRA 后端。
- **MPS**：Apple Silicon 本地 LoRA 训练后端，覆盖模型注入、外部 Trainer
  训练、adapter I/O、评估和 merge/export。

CPU 保留为功能回退和单元测试环境，不作为大模型训练目标。

## 精度策略

| 后端 | `auto` 默认 | 可选精度 | 说明 |
|---|---|---|---|
| CUDA（支持 BF16） | BF16 | FP32、FP16、BF16 | BF16 优先 |
| CUDA（不支持 BF16） | FP16 | FP32、FP16 | 适合较旧 GPU |
| MPS | FP32 | FP32、FP16 | FP32 稳定优先，FP16 降低内存 |
| CPU | FP32 | FP32 | 功能验证 |

统一 Demo 可以显式选择：

```bash
# MPS 稳定默认
python examples/lora_sft_demo.py train --dtype auto

# MPS 低内存模式；应先用短训练检查 loss 和梯度有限性
python examples/lora_sft_demo.py train --dtype float16 --max-steps 20
```

MPS FP16 是否稳定取决于模型、算子、optimizer 和 PyTorch 版本。如果出现
NaN、Inf 或 loss 异常，应切回 FP32。

## 当前能力边界

CUDA 与 MPS 当前都支持普通 `nn.Linear` LoRA 闭环。NanoFT 还提供实验性
QLoRA 配置、bitsandbytes 4-bit 包装和反量化合并，但只在 CUDA +
bitsandbytes 路径上工作；需要先安装 `nanoft[qlora]`。MPS 量化训练不会复用
不兼容的 CUDA 实现，也不会宣称已经支持。

NanoFT 不计划为 MPS 重写 MLX Trainer。外部训练仍由 TRL、Transformers
或 PyTorch 执行。
