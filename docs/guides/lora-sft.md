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

默认最多读取 2048 个样本、batch size 1、gradient accumulation 4，
`--max-steps` 默认为 100。使用满 2048 条时，一轮约 512 step；设置
`--max-steps -1 --num-train-epochs 1` 可按一轮训练，`--max-samples 0`
可读取训练文件的全部记录。根据验证 loss 与任务正确性选择 checkpoint，
不要仅通过增加 step 数追求更低 loss。

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

### BF16 基座与 FP32 adapter 参数

使用 Python API 时，可以先将基座加载为 BF16，再显式指定 dense LoRA 的
参数精度：

```python
import torch
from nanoft import LoRAConfig, prepare_model_for_training

model = prepare_model_for_training(
    base_model,  # 已加载为 torch.bfloat16
    LoRAConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
    adapter_dtype=torch.float32,
)
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=2e-4,
)
```

应先设置 adapter dtype，再构造 optimizer。此设置保留 BF16 基座，使 LoRA
A/B 参数、梯度以及标准 AdamW 的动量状态为 FP32；外部训练框架的 BF16
autocast 仍可以在矩阵计算中使用 BF16。参数保存精度与计算精度是独立选择，
该选项不会关闭 autocast。注入后对整个模型调用 `.to(dtype=...)` 也会转换
adapter 参数，因此应在注入前设置基座精度。

不传 `adapter_dtype` 时保持原有行为，adapter 跟随基座权重 dtype。
该参数仅支持 dense LoRA，QLoRA 继续使用其量化配置中的 `compute_dtype`。
保存 FP32 adapter 后，重新加载也需显式选择 FP32，见
[Adapter I/O](../api/adapter-io.md#load_adapter)。

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

### Prompt/completion 数据与问题分组

新任务可使用已经按目标模型 chat template 格式化的两个字符串列：

```json
{"prompt": "<包含用户问题及助手回答起始标记的完整前缀>", "completion": "<助手回答正文及模型要求的结束标记>"}
```

`prompt` 和 `completion` 都必须是非空字符串。它们不是 messages 数组，
示例不会自动转换原始 `instruction`、Alpaca 或 ShareGPT 数据，也不处理
图像、音频。生成前缀应与后续推理使用的模板相同。

先按完整 prompt 分组，再划分训练与验证：

```bash
python examples/split_sft_data.py \
  --input data/formatted.jsonl \
  --output-dir data/grouped \
  --validation-size 2048 \
  --seed 3407
```

输出为 `train.jsonl`、`validation.jsonl` 和 `manifest.json`。输出目录必须
不存在；`--validation-size` 是目标行数，整组保留可能使实际验证行数更多。
至少需要两个不同 prompt，并且必须能保留非空训练集。

工具按原始 prompt 字符串的 SHA256 分组，不 strip、不改大小写或规范化
空白；同组的全部答案及重复记录保留在同一侧，其他字段也原样保留。
清单记录 seed、源/输出文件 SHA256、实际行数、组 ID 与源行号，便于复核
prompt 精确交集为零。这不能排除改写问题、语义重合或基座预训练接触。

训练和独立评估都显式选择同一监督范围：

```bash
python examples/lora_sft_demo.py train \
  --model-name /path/to/model \
  --data-path data/grouped/train.jsonl \
  --eval-data-path data/grouped/validation.jsonl \
  --data-format prompt-completion --completion-only-loss \
  --dtype bfloat16 --adapter-dtype float32 \
  --amp auto --full-determinism --attn-implementation eager \
  --max-samples 0 --max-steps -1 --num-train-epochs 1 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --output-root outputs/my_run
```

该命令展示一轮训练的参数组合，实际运行前仍应先用少量样本做 smoke test。
`--completion-only-loss` 屏蔽 prompt labels，保留 completion 中可用于 causal
loss 的 token；`--no-completion-only-loss` 监督完整序列，也是默认值。
`--data-format text --completion-only-loss` 会明确拒绝。

Prompt/completion 先联合分词，且不会再次自动添加 special tokens；缺失末尾
EOS 时追加 tokenizer 的 EOS。若单独 prompt 的 tokens 不是联合分词结果的
前缀，或右侧截断后没有有效监督 token，会报错，需修正分界或长度。
旧 `text` 模式保留 tokenizer 默认 special-token 行为，因此已有 chat-formatted
数据仍需检查 BOS/EOS，不能假定两种格式处理完全相同。

### 统一训练、评估与合并的执行条件

以下参数在 `train`、`evaluate`、`merge` 中共用：

| 参数 | 默认值及含义 |
| --- | --- |
| `--dtype auto` | 基座精度：CUDA 选择 BF16/FP16，MPS/CPU 选择 FP32 |
| `--adapter-dtype auto` | 训练跟随基座；独立加载时读取 checkpoint 中唯一的 LoRA dtype。可显式选 `float32`、`float16`、`bfloat16` |
| `--amp auto` | CUDA BF16/FP16 使用 autocast；`off` 关闭。FP32 主参数不等于 FP32 矩阵计算 |
| `--full-determinism` | 默认关闭；启用后统一使用 Transformers 的确定性执行设置 |
| `--attn-implementation auto` | 保留模型默认选择；可显式选 `eager` 或 `sdpa` |
| `--seed 3407` | 各阶段使用同一随机种子 |

CLI 的 checkpoint dtype 检测不同于核心 `load_adapter()` 默认行为；后者的
显式 FP32 加载见 [Adapter I/O](../api/adapter-io.md#load_adapter)。CLI 遇到
混合或无法识别的 checkpoint dtype 时，`auto` 会拒绝猜测。

做数值对照时，各阶段应使用相同参数，并比较产物中记录的实际执行配置，
包括参数 dtype、AMP、attention、确定性、TF32 和环境变量。训练记录还包含
Trainer 初始化后的实际 mixed precision。相同 seed 本身不能替代这些检查。
这些选项用于控制实验条件，不保证微调效果改善或跨硬件逐位一致。

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

评估结果包括按有效监督 token 加权的 loss、perplexity、生成及实际执行配置，
写入 `evaluation_results.json`。默认 `--eval-offset 2048` 只是行偏移，
不是独立划分保证。新数据使用上面的分组验证文件，并显式设为 offset 0：

```bash
python examples/lora_sft_demo.py evaluate \
  --model-name /path/to/model \
  --data-path data/grouped/validation.jsonl \
  --data-format prompt-completion --completion-only-loss \
  --output-root outputs/my_run --variants base,native,peft \
  --dtype bfloat16 --adapter-dtype float32 \
  --amp auto --full-determinism --attn-implementation eager \
  --eval-offset 0 --eval-samples 64 --eval-max-length 2048
```

比较各模型时保持验证数据、截断长度和监督 mask 相同。训练 loss 沿用
TRL/Transformers 的聚合口径，不能与独立 token 加权 loss 直接当作同一数值。
独立评估现在与训练一样默认启用 CUDA autocast、追加缺失 EOS；与历史不使用
AMP 或 EOS 处理不同的评估记录比较时，应明确标注口径变化。

默认数据使用 `<human>/<bot>` prompt。对于其他数据格式，可用
`--prompt-template` 传入包含 `{prompt}` 的模板，或通过 `--prompts-file`
提供 JSONL 测试问题。

上面非默认模型的评估命令还应配合该模型的 `--prompt-template`，否则默认
OIG 生成提示与训练模板不一致。验证集 loss 的数据格式与自由生成提示模板
是两项独立输入。固定任务题集与评分标准，再比较原模型和微调模型的实际答案。
生成时，`prompt-completion` 模式不重复添加 special tokens，`text` 模式保留
原行为；`evaluation_results.json` 的 `generation_add_special_tokens`
记录实际选择。因此 prompt/completion 模式的生成模板必须包含模型需要的前缀。

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
  --output-dir outputs/my_run/merged_model \
  --dtype bfloat16 --adapter-dtype float32 \
  --amp auto --full-determinism --attn-implementation eager
```

脚本会分别报告：

- adapter model 到内存中 merged model 的低精度 merge 漂移，仅作诊断；
- 内存中 merged model 到重新加载产物的一致性，决定序列化校验是否通过。

合并目录还写出 `execution_config.json` 与 `merge_verification.json`，
分别记录执行条件、合并前后参数 dtype 及两项数值比较。合并验证使用普通
文本检查提示，其 `verification_add_special_tokens` 与聊天生成的设置
单独记录；重载验证通过不等于合并前后所有回答都相同。
