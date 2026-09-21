# 核心概念与边界

## NanoFT 是模型准备工具，不是 Trainer

NanoFT 处理“训练什么”和“如何保存”：

- LoRA 层实现与目标模块替换。
- 参数冻结和 adapter 参数开放。
- adapter 配置、权重格式、保存与加载。
- LoRA merge/unload 和标准模型导出。
- 数据格式准备与训练框架集成表面（逐步建设）。

NanoFT 不处理“怎样跑训练任务”：

- 完整训练循环。
- optimizer、scheduler 和 checkpoint 策略。
- 分布式训练、DeepSpeed、FSDP 或集群调度。
- 日志、实验跟踪和长期训练状态。
- 对 `SFTTrainer`、`DPOTrainer` 等 Trainer 的二次封装。

## LoRA 的作用

对于 Linear 权重 `W`，NanoFT 训练低秩矩阵 `A` 和 `B`：

```text
y = xWᵀ + scaling × xAᵀBᵀ
scaling = lora_alpha / r
```

base 权重保持冻结，通常只有很少一部分参数参与优化。训练后可以保留
adapter，也可以把 `B @ A` 合并回 base 权重。

## Adapter 与完整模型

Adapter 只包含 LoRA 权重与配置，体积小，但加载时仍需要与训练时相同的
base model。Merged model 包含完整权重，体积大，但推理时不需要 NanoFT
或 PEFT。

## 当前稳定范围

NanoFT 0.2.0 稳定支持普通 `nn.Linear` 上的 LoRA。当前边界包括：

- `fan_in_fan_out=True`。
- 对同一模型重复注入 LoRA。
- 在普通 merge API 中静默处理量化权重。
- QLoRA 目前是实验性能力，只支持 CUDA + bitsandbytes；完整训练闭环仍需
  在目标硬件和外部 Trainer 上验证。
- 内置 SFT、DPO 或在线 RL Trainer。

QLoRA 使用独立配置组合 LoRA 与量化选项；preference/RL 方向只提供
模型、adapter 与数据准备能力，不进入训练编排。
