# NanoFT

**Nano Fine-Tuning** — 轻量级端侧大模型微调框架

NanoFT 是一个面向边缘设备（Jetson GPU、Apple MPS、ARM CPU）的 LLM 微调框架，纯 Python 实现，与 HuggingFace 生态兼容。

## 特性

- **LoRA 微调** — 低秩适配器，只训练极少量参数
- **QLoRA** — NF4 量化训练，显存降低 4 倍（开发中）
- **多设备支持** — CUDA / MPS / CPU 自动检测
- **独立 Adapter 格式** — 默认保存 NanoFT 原生 adapter，可显式导出 PEFT 兼容格式
- **推理加速** — 权重合并 + GGUF/ONNX 导出（开发中）
- **轻量依赖** — 仅需 torch、transformers、datasets、safetensors

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
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

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

# ... 用 HuggingFace Trainer 训练 ...

# 保存 adapter（NanoFT 原生格式）
save_adapter(model, "./adapter", config)

# 如需 PEFT 兼容格式，显式导出
save_peft_adapter(model, "./adapter_peft", config)

# 合并权重并导出完整模型；保存前会自动卸载 LoRA 层
save_merged_model(model, "./merged_model", tokenizer=tokenizer)
```

完整训练示例见 [examples/train_lora.py](../examples/train_lora.py)。

## 配置说明

```python
config = LoRAConfig(
    r=8,                    # LoRA 秩
    lora_alpha=32,          # 缩放因子，实际 scaling = alpha / r = 4.0
    lora_dropout=0.1,       # Dropout 概率
    target_modules=[        # 要替换的层（子串匹配）
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
| `apply_lora(model, config)` | 注入 LoRA 层（不冻结参数） |
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
nanoft/
├── config.py          # LoRAConfig 配置
├── layers.py          # LoRALinear 层实现
├── apply.py           # LoRA 注入/移除
├── merge.py           # 权重合并
├── save_load.py       # Adapter 保存/加载（NanoFT 原生 + PEFT 兼容）
├── device.py          # 设备检测
├── train_utils.py     # 训练准备工具
├── export/            # GGUF / ONNX 导出（开发中）
└── engine/            # 内置推理引擎（开发中）
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

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B")
model = PeftModel.from_pretrained(base_model, "./adapter_peft")
```

## MPS 训练

在 Apple MPS 上，如果 fp16 训练出现 `nan` 梯度或非有限权重，可以在训练配置里显式关闭混合精度：

```json
{
  "training": {
    "fp16": false,
    "bf16": false
  }
}
```

## 依赖

```
# 必须
torch>=2.8.0
transformers>=4.36.0
datasets>=2.14.0
safetensors>=0.4.0

# 可选
gguf>=0.6.0          # GGUF 导出
onnx>=1.15.0         # ONNX 导出
```

## License

MIT
