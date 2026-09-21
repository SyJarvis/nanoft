# NanoFT

NanoFT 是一个轻量级大模型微调框架与工具集。它负责 LoRA 层、adapter
注入与冻结、adapter I/O、权重合并和训练框架衔接，同时把训练循环、
optimizer、scheduler、分布式执行和实验管理留给 TRL、Transformers、
PyTorch 或 Accelerate。

## 当前能力

- 将 Hugging Face 或普通 PyTorch 模型中的目标 `nn.Linear` 替换为
  `LoRALinear`。
- 冻结 base model，只开放 LoRA 参数以及配置指定的 bias。
- 保存 NanoFT 原生 adapter 或显式导出 PEFT 兼容 adapter。
- 从 NanoFT/PEFT adapter 恢复模型。
- 合并并卸载 LoRA，导出标准 Transformers 模型。
- 检测 CUDA、Apple MPS 和 CPU 环境。
- 将准备好的模型交给外部 Trainer，不接管训练循环。

CUDA（NVIDIA/Jetson）和 MPS（Apple Silicon）是 NanoFT 的正式端侧 GPU
后端；CPU 用于功能回退。具体精度与能力边界见
[CUDA 与 MPS 支持](getting-started/device-support.md)。

{% hint style="info" %}
NanoFT 0.2.0 的稳定闭环是普通 LoRA。当前版本还提供实验性的 QLoRA 配置、
bitsandbytes 量化包装和反量化合并 API；它只面向 CUDA + bitsandbytes，不能
视为 MPS 支持。SFT 数据规范化和 preference/RL 数据准备仍属于后续规划。
{% endhint %}

## 从这里开始

1. [安装 NanoFT](getting-started/installation.md)
2. [完成第一个 LoRA adapter](getting-started/quickstart.md)
3. [理解 NanoFT 的职责边界](getting-started/concepts.md)
4. [运行统一 LoRA SFT Demo](guides/lora-sft.md)

## 核心工作流

```text
Transformers/PyTorch model
          │
          ▼
LoRAConfig + prepare_model_for_training()
          │
          ▼
TRL / Transformers / PyTorch 执行训练
          │
          ├── save_adapter() ──────── NanoFT adapter
          ├── save_peft_adapter() ─── PEFT adapter
          └── save_merged_model() ─── 标准完整模型
```

## 项目状态

NanoFT 仍处于 Alpha 阶段。公开能力、未来范围和不计划承担的训练框架职责，
分别记录在 [核心概念](getting-started/concepts.md) 与
[Roadmap](development/roadmap.md) 中。
