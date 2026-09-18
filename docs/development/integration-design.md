# 训练框架集成设计

本文档最初基于 Unsloth 对照实现整理，现在由统一 demo 承载 NanoFT 工作流：

- `benchmarks/unsloth_qwen25_05b_chip2.py`
- `examples/lora_sft_demo.py`

目标不是复制 Unsloth 的内部实现，而是匹配其顺畅的用户工作流：加载模型 → 接 LoRA → 准备数据 → 交给训练框架 → 保存 adapter 和合并后的模型。

## 基于 Demo 的工作流

当前 NanoFT + TRL 的 demo 有六个用户可见阶段：

1. 加载 JSONL 文本数据集并选择训练样本。
2. 加载 HuggingFace tokenizer 和模型。
3. 构造 `LoRAConfig` 并通过 `prepare_model_for_training()` 注入 LoRA。
4. 把模型和数据集交给 TRL `SFTTrainer`。
5. 通过 TRL 执行训练，使用独立子命令评估 held-out 数据。
6. 保存 NanoFT adapter、PEFT adapter、合并模型和指标。

NanoFT 应将第 4、5 阶段保留在核心之外。第 1、2、3、6 阶段是 NanoFT helper 的合适范围——它们为训练框架准备输入和输出。

## Demo 暴露的 API 缺口

### 模型加载

当前 demo 代码：

```python
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)
```

需要的 NanoFT helper：

```python
model, tokenizer = nanoft.load_hf_causal_lm(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    ensure_pad_token=True,
)
```

范围：

- 对 Transformers 的薄包装，集中常见 kwargs（`device_map`、`trust_remote_code`、dtype、pad_token）。
- 缺失时设置 `tokenizer.pad_token`。
- **必须返回标准 HuggingFace 对象**（`AutoModelForCausalLM` / `PreTrainedTokenizer`），不返回任何 NanoFT 子类。
- **必须可被绕过**：用户用 `AutoModelForCausalLM.from_pretrained(...)` 自行加载（量化、自定义 device_map、私有 model class 等），再喂给 `prepare_model_for_training(model, config)` 与 helper 路径完全等价。
- 通过 `**from_pretrained_kwargs` 透传罕见参数（`revision`、`token`、`attn_implementation` 等），不假装能枚举所有 HF 参数。

> 设计原则：NanoFT 的 helper 只做"集中样板"，不做"锁人"。返回标准类型 + 下游签名接受标准类型（`nn.Module`、`PreTrainedTokenizer`），两条同时成立时 helper 才是可选便捷，不是 Unsloth 式陷阱。

### LoRA 准备

当前 demo 代码**已是正确设计，不需要替换**：

```python
lora_config = LoRAConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=target_modules,
    bias="none",
    base_model_name_or_path=model_name,
)
model = prepare_model_for_training(model, lora_config)
```

这一两步式 API 满足核心原则：

- **config 可见且用户拥有**：`LoRAConfig` 显式构造，可序列化为 PEFT 格式。
- **应用与构造分离**：`prepare_model_for_training(model, config)` 接收 config 作为输入，不在签名里复制 config 字段。
- **不隐藏训练框架**：函数只做 freeze + attach，不创建 trainer。

**不引入** `prepare_lora_model(model, r=8, ...)` 这类合并函数。那样会把 `LoRAConfig` 的字段重新搬到 apply 函数签名里，造成两份参数表必须同步漂移，并削弱 config 作为单一真相源——这是 Unsloth `get_peft_model` 的反模式。

**不引入** `target_modules` 预设常量（如 `presets.QWEN2_FULL` 或字符串 `"qwen2_full"`）。直接传列表即可，多一层抽象只会过时、重复，且让用户必须查常量含义而不是直接看值。

### 数据集准备

当前 demo 代码：

```python
dataset = load_dataset("json", data_files={"train": data_path}, split="train")
dataset = dataset.select(range(min(max_samples, len(dataset))))
split = dataset.train_test_split(test_size=0.02, seed=42)
```

需要的 NanoFT helper：

```python
split = nanoft.data.load_sft_dataset(
    data_path,
    format="text",
    text_column="text",
    max_samples=2048,
    test_size=0.02,
    seed=42,
)
```

范围：

- 归一化常见 SFT 格式：text、prompt/completion、Alpaca、ShareGPT、OpenAI messages。
- 返回标准 HuggingFace `DatasetDict` 或 split dict。
- 避免训练框架相关行为，除非放在 integration 模块下。

### TRL 集成

**不封装。** TRL 的 `SFTTrainer` + `SFTConfig` 已经是对 Transformers `Trainer` 的方便层，再叠一层 NanoFT wrapper 只会复制 `SFTConfig` 的参数表（`max_length`、`dataset_text_field`、`output_dir` 等），两份参数同步漂移，且 `training_kwargs={...}` 这种逃生舱自己就承认覆盖不全。

NanoFT 的职责是给训练框架**准备输入和输出**，不是替代训练框架。如何把 NanoFT 准备好的 `model` 和 `dataset` 接入 TRL，由 `examples/lora_sft_demo.py train` 展示，原始 TRL 代码直接可用，无需 helper。

### 输出保存

当前 demo 代码：

```python
save_adapter(model, adapter_dir, lora_config)
save_peft_adapter(model, peft_adapter_dir, lora_config)
save_merged_model(model, merged_dir, tokenizer=tokenizer, inplace=False)
```

需要的 NanoFT helper：

```python
nanoft.save_finetuned_outputs(
    model,
    tokenizer=tokenizer,
    config=lora_config,
    output_dir=output_dir,
    save_native_adapter=True,
    save_peft_adapter=True,
    save_merged=True,
)
```

范围：

- 标准化输出目录结构。
- 可选写入运行元信息。
- 保留底层单个 save 函数供高级用户使用。

> 设计注：若未来输出种类继续增长（optimizer state、training args、运行元信息等），
> 布尔标志会膨胀。届时可考虑用 `SaveConfig` dataclass 替代多个 bool，
> 每个字段是子 options dataclass（`None` = 不保存）。
> 当前三 bool 形式足够，无需提前抽象。

## 提议的 v0.2 包结构

```text
src/nanoft/
├── models.py              # load_hf_causal_lm, tokenizer 工具
├── data/
│   ├── __init__.py
│   ├── schemas.py         # 归一化的 SFT/DPO 样本 schema
│   ├── loaders.py         # load_sft_dataset
│   └── converters.py      # Alpaca/ShareGPT/messages/text 转换
└── outputs.py             # save_finetuned_outputs, metadata helpers
```

不设 `integrations/` 模块——TRL、Transformers Trainer 等训练框架由 demo 脚本展示接入方式，不在 NanoFT 内做 wrapper。

## 需要的文档

NanoFT 应提供面向迁移的文档：

- 从原始 Transformers Trainer 迁移到 NanoFT adapter 准备。
- 从 PEFT LoRA 注入迁移到 NanoFT LoRA 注入。
- 从 Unsloth 风格的 SFT 脚本迁移到 NanoFT + TRL 脚本。
- 如何保存 NanoFT-native adapter、PEFT-compatible adapter 和合并后的 HuggingFace 模型。

每篇指南应同时展示 NanoFT helper 的简短路径和等价的原始训练框架代码，
让用户能在不放弃对训练栈控制权的前提下采用 NanoFT。
