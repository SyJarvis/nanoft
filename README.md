# NanoFT

**Nano Fine-Tuning** — 轻量级端侧大模型微调框架与工具集

NanoFT 是一个面向边缘设备（Jetson GPU、Apple MPS、ARM CPU）的 LLM 微调框架与工具集，纯 Python 实现，与 HuggingFace 生态兼容。它专注于微调方法实现、模型/adapter 改造、数据处理、格式转换和训练框架适配；训练执行由 TRL、Transformers、PyTorch、Accelerate 等成熟框架负责。

## 定位

NanoFT 保持轻量，不重新实现训练框架。它提供可组合的微调组件、迁移文档和基础集成案例，让用户可以把 NanoFT 快速接入自己的 TRL、Transformers、PyTorch 或 Accelerate 训练代码。

NanoFT 负责：

- **微调方法实现** — LoRA/QLoRA、adapter 注入、参数冻结、权重合并与卸载
- **模型与 adapter I/O** — NanoFT 原生 adapter、PEFT 兼容导出、base model + adapter 加载
- **数据处理与格式转换** — 面向 SFT/DPO/chat template 的数据规范化、prompt 格式化、tokenization 前处理
- **训练框架适配** — 产出可直接交给 TRL、Transformers、PyTorch 使用的模型、数据集或配置，并提供迁移文档与基础案例
- **推理与验证工具** — 合并权重、快速加载 adapter 做本地验证

NanoFT 不负责：

- 完整训练循环或自定义 Trainer 抽象
- 分布式训练编排、DeepSpeed/FSDP/Accelerate 调度
- optimizer、scheduler、checkpoint、日志和实验平台的完整封装
- 重新实现 `transformers.Trainer`、`trl.SFTTrainer`、`trl.DPOTrainer`

## 特性

- **LoRA adapter** — 低秩适配器，只训练极少量参数
- **QLoRA 准备能力** — CUDA + bitsandbytes NF4 量化支持（实验性）
- **多设备支持** — CUDA / MPS / CPU 自动检测
- **独立 Adapter 格式** — 默认保存 NanoFT 原生 adapter，可显式导出 PEFT 兼容格式
- **推理加速准备** — LoRA 权重合并到 base model，导出无 adapter 依赖的标准模型
- **训练框架友好** — 提供 TRL、Transformers、PyTorch 等常见框架的衔接示例
- **轻量依赖** — 仅需 torch、transformers、datasets、safetensors

## 路线图

NanoFT 的总体路线、v0.1 稳定目标和 v0.2 发展重点见
[docs/development/roadmap.md](docs/development/roadmap.md)。完整文档使用
GitBook 结构组织，入口见 [docs/README.md](docs/README.md)。

## 安装

```bash
pip install -e .
```

## 快速开始

```python
from nanoft import (
    LoRAConfig,
    prepare_model_for_training,
    print_trainable_parameters,
    save_adapter,
    save_peft_adapter,
    save_merged_model,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 配置 LoRA
config = LoRAConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
)

# 注入 LoRA 层
model = prepare_model_for_training(model, config)
print_trainable_parameters(model)
# Trainable parameters: 1,572,864 / 2,154,738,688 (0.07%)

# ... 将 model 接入 HuggingFace Trainer / TRL / PyTorch 训练代码 ...

# 保存 adapter（NanoFT 原生格式）
save_adapter(model, "./adapter", config)

# 如需 PEFT 兼容格式，显式导出
save_peft_adapter(model, "./adapter_peft", config)

# 合并权重并导出完整模型；默认通过副本保存，不修改当前 LoRA 模型
save_merged_model(model, "./merged_model", tokenizer=tokenizer)
```

统一训练框架衔接示例见
[examples/lora_sft_demo.py](examples/lora_sft_demo.py)。它通过 `train`、
`evaluate` 和 `merge` 三个子命令展示 NanoFT + TRL 的完整 LoRA 工作流；
Qwen3-0.6B 只是默认演示模型，模型与目标层均可替换。

完整使用说明见
[docs/guides/lora-sft.md](docs/guides/lora-sft.md)。

## 配置说明

```python
config = LoRAConfig(
    r=8,                    # LoRA 秩
    lora_alpha=32,          # 缩放因子，实际 scaling = alpha / r = 4.0
    lora_dropout=0.1,       # Dropout 概率
    target_modules=[        # 要替换的层（模块名精确或后缀匹配）
        "q_proj", "k_proj", "v_proj"
    ],
    bias="none",            # "none" | "all" | "lora_only"
    task_type="CAUSAL_LM",  # 任务类型
)
```

## API 参考

### 核心接口

| 函数 | 说明 |
|------|------|
| `prepare_model_for_training(model, config)` | 冻结参数 + 注入 LoRA 层 |
| `apply_lora(model, config)` | 注入 LoRA 层并冻结 base 参数 |
| `remove_lora(model)` | 移除 LoRA，还原为标准 Linear |
| `merge_lora_weights(model)` | 合并 LoRA 到原权重（推理加速） |
| `merge_and_unload_lora(model)` | 合并 LoRA 并替换回普通 Linear |
| `save_adapter(model, path, config)` | 保存 NanoFT 原生 adapter |
| `save_peft_adapter(model, path, config)` | 保存 PEFT 兼容 adapter |
| `load_adapter(model, path)` | 加载 NanoFT 或 PEFT adapter |
| `save_merged_model(model, path)` | 保存无 LoRA 参数残留的完整模型 |
| `detect_device()` | 检测可用设备 |
| `print_trainable_parameters(model)` | 打印可训练参数统计 |

### 层类型

| 类 | 说明 |
|----|------|
| `LoRALinear` | 带 LoRA 的线性层，继承 `nn.Linear` |
| `LoRAConfig` | 配置 dataclass，支持 PEFT 序列化 |

## 项目结构

```
src/nanoft/
├── config.py          # LoRAConfig 配置
├── layers.py          # LoRALinear 层实现
├── apply.py           # LoRA 注入/移除
├── merge.py           # 权重合并
├── save_load.py       # Adapter 保存/加载（NanoFT 原生 + PEFT 兼容）
├── device.py          # 设备检测
├── quant.py           # QLoRA 量化包装与反量化合并
└── train_utils.py     # 模型训练前准备，不实现训练循环
```

## Adapter 格式

NanoFT 默认输出原生 adapter，权重 key 与模型内部 `state_dict` 一致：

```
adapter/
├── adapter_config.json
└── adapter_model.safetensors    # q_proj.lora_A / q_proj.lora_B 等内部 key
```

`load_adapter()` 会自动识别 NanoFT 原生格式和 PEFT 兼容格式。

如需给 PEFT 直接加载，使用 `save_peft_adapter()`：

```python
from nanoft import save_peft_adapter
from peft import PeftModel

save_peft_adapter(model, "./adapter_peft", config)

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(base_model, "./adapter_peft")
```

## CUDA 与 MPS

NanoFT 正式支持 CUDA（包括 Jetson）和 Apple MPS 上的普通 LoRA 闭环。
统一 Demo 在 CUDA 上自动选择 BF16/FP16，在 MPS 上默认使用 FP32：

```bash
python examples/lora_sft_demo.py train --dtype auto
```

Apple Silicon 内存受限时可以显式使用 `--dtype float16`，但应先运行短训练
确认 loss、梯度和 adapter 权重保持有限。QLoRA 首期以 CUDA backend 为主，
MPS 量化训练等待独立、可验证的后端实现。

## 依赖

核心安装只包含模型准备和 adapter I/O 所需依赖：

```bash
pip install -e .
```

运行统一 TRL SFT 示例时安装训练框架扩展：

```bash
pip install -e '.[trl]'
```

如需 PEFT 互操作或实验性 QLoRA：

```bash
pip install -e '.[peft]'
# QLoRA 还要求 CUDA；bitsandbytes 的平台支持以其发行说明为准
pip install -e '.[qlora]'
```

对应的核心依赖为：

```
torch>=2.8.0
transformers>=4.51.0
datasets>=2.14.0
safetensors>=0.4.0
```

## License

MIT
