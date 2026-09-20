# 远端 Gemma-4-E2B-it LoRA SFT 操作指南

本文针对 SSH 主机 `203`（`prod-toolchain-01`），使用 Docker 容器
`zrk_worker`（镜像 `gpu-train:cuda12.8-torch2.8`）和项目目录
`/workspace/nanoft`，在本地模型
`/mnt/data8t/models/google--gemma-4-E2B-it/snapshots/master/` 上运行
NanoFT 的 LoRA SFT Demo。

本文只给出远端执行命令，不代替远端执行安装或数据下载。当前远端环境已知
`trl` 和 `peft` 已安装。训练命令需要 `[trl]`；只有运行 PEFT 互操作或
`peft` variant 评估时才需要 `[peft]`。为完整执行本文的评估流程，安装命令
仍一次性包含 `.[trl,peft]`。QLoRA 不在本文训练流程中。

文件名沿用早期 Qwen 计划，本文实际验证对象是上述 Gemma/203 环境。
**全量文本 LoRA 已完成，效果回归仍需处理。** 下方保留当天早期 `text`
格式命令及实测记录用于追溯；新任务使用本节的分组数据和共同执行参数，
不要把旧数据划分或低 loss 当作泛化验收。

## 当前推荐流程

先完成下方连接、环境检查。新数据应为已使用 Gemma tokenizer 格式化的
`prompt` / `completion` 字符串；字段边界、EOS 和监督规则见
[SFT 数据说明](lora-sft.md#promptcompletion-数据与问题分组)。旧转换脚本只
输出 `text`，不能直接作为分组工具的输入。`PAIR_DATA` 需由操作者准备，
下面不假定它已存在。新输出目录也应与历史运行分开：

```bash
cd /workspace/nanoft
export MODEL_PATH=/mnt/data8t/models/google--gemma-4-E2B-it/snapshots/master/
export PAIR_DATA=/workspace/data/gemma4_prompt_completion.jsonl
export GROUPED_DATA=/mnt/data8t/nanoft-runs/gemma4-next-data
export NEXT_RUN=/mnt/data8t/nanoft-runs/gemma4-next-smoke
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
test -s "$PAIR_DATA"

python examples/split_sft_data.py \
  --input "$PAIR_DATA" --output-dir "$GROUPED_DATA" \
  --validation-size 2048 --seed 3407

execution_args=(
  --model-name "$MODEL_PATH" --dtype bfloat16 --adapter-dtype float32
  --amp auto --full-determinism --attn-implementation eager --seed 3407
)
data_args=(--data-format prompt-completion --completion-only-loss)

python examples/lora_sft_demo.py train \
  "${execution_args[@]}" "${data_args[@]}" \
  --data-path "$GROUPED_DATA/train.jsonl" \
  --eval-data-path "$GROUPED_DATA/validation.jsonl" --eval-steps 20 \
  --target-modules "$TARGET_MODULES" --output-root "$NEXT_RUN" \
  --max-samples 64 --max-steps 20 --max-seq-length 512 \
  --batch-size 1 --gradient-accumulation-steps 1 --save-steps 20
```

分组工具要求输出目录不存在，保留每个完整 prompt 的全部记录；目标验证行数
可能因整组保留而超出。它不清洗数据、不规范化空白，也不保证语义或预训练
无泄漏。先审查 `manifest.json` 和具体样本，再决定训练范围。

评估前，使用下方“评估 adapter”节的 tokenizer 脚本设置 `PROMPT_TEMPLATE`，
保留 Gemma 的全部生成前缀。在同一 Bash session 中继续运行：

```bash
test -n "${PROMPT_TEMPLATE:-}"
python examples/lora_sft_demo.py evaluate \
  "${execution_args[@]}" "${data_args[@]}" \
  --data-path "$GROUPED_DATA/validation.jsonl" --eval-offset 0 \
  --eval-samples 64 --eval-max-length 512 \
  --output-root "$NEXT_RUN" --variants base,native,peft \
  --prompt-template "$PROMPT_TEMPLATE" --max-new-tokens 128

python examples/lora_sft_demo.py merge \
  "${execution_args[@]}" \
  --adapter "$NEXT_RUN/adapter_native" \
  --output-dir "$NEXT_RUN/merged_model" --verify
```

`--adapter-dtype float32` 指 A/B 主参数；`--amp auto` 仍在 CUDA 上使用 BF16
autocast。训练、评估与合并统一这些参数，并核对记录中的实际执行条件，
不能只比较命令行或 seed。CLI 的 `--adapter-dtype auto` 在加载时会读取
checkpoint 的唯一 LoRA dtype，避免把 FP32 adapter 静默加载成 BF16。
当前独立评估也使用 AMP 和统一的 EOS/labels 规则，不能与旧评估口径混算。

这组参数用于可复核的文本训练，并未证明能解决历史回归。smoke 通过后，
仍应先用目标任务验收回答质量，再考虑扩大数据和步数。图像、音频和真实
CUDA/bitsandbytes QLoRA 均不在本文已验证范围。

## 连接和环境检查

SSH 连接：

```bash
ssh 203
```

登录后直接操作容器（`docker exec` 默认以 root 进入）：

```bash
docker exec -it zrk_worker bash
```

或在宿主机上通过 `docker exec zrk_worker <command>` 执行单条命令。

先执行环境检查。它只读取版本和 GPU 信息，不会安装软件或下载数据：

```bash
hostname
python --version
which python
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

python - <<'PY'
from importlib.util import find_spec

for name in ("torch", "transformers", "datasets", "safetensors", "trl", "peft", "bitsandbytes"):
    spec = find_spec(name)
    if spec is None:
        print(f"{name}: MISSING")
        continue
    try:
        module = __import__(name)
    except Exception as exc:
        print(f"{name}: IMPORT FAILED ({type(exc).__name__}: {exc})")
    else:
        print(f"{name}: {getattr(module, '__version__', 'version unavailable')}")

import torch

print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

继续之前应确认：`torch.cuda.is_available` 为 `True`，目标 GPU 在
`nvidia-smi` 中有足够显存，`datasets`、`transformers` 和 `safetensors`
可导入。TRL 和 PEFT 应已安装并可导入。`bitsandbytes` 当前缺失，但本文的
普通 LoRA 路径不会调用它。

## 安装项目和训练扩展

在确认项目路径正确后，由远端操作者执行（通常已安装，只需确认）：

```bash
cd /workspace/nanoft
python -m pip install -e '.[trl,peft]'
```

这个 extras 会安装 NanoFT 的可编辑版本、TRL/Accelerate 和 PEFT。训练本身
最低只需要 `.[trl]`；本文安装 `.[trl,peft]` 是为了后续可以直接运行
`peft` variant 的互操作评估。安装完成后再次运行上一节的导入检查，并确认
当前包来自工作树：

```bash
python -c "import nanoft; print(nanoft.__version__, nanoft.__file__)"
```

如果输出的 `nanoft.__file__` 不在 `/workspace/nanoft` 下，先修正安装，
再继续训练。

## 选择 GPU 和运行参数

Demo 没有单独的 `--device` 参数；它通过 PyTorch 自动检测 CUDA。当前容器
只有一张 RTX 5090（32 GB），无需设置 `CUDA_VISIBLE_DEVICES`。
如果先前把该变量设为空字符串，执行 `unset CUDA_VISIBLE_DEVICES` 恢复
容器内的 GPU 可见性；空字符串会屏蔽全部 GPU。

定义本次运行的路径。`DATA_PATH` 必须指向远端已经存在的 JSON/JSONL 文件；
不要把它替换成需要下载的 URL：

```bash
export MODEL_PATH=/mnt/data8t/models/google--gemma-4-E2B-it/snapshots/master/
export DATA_PATH=/workspace/data/unified_chip2.jsonl
export OUTPUT_ROOT=/workspace/outputs/gemma4-sft
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

test -d "$MODEL_PATH"
test -s "$DATA_PATH"
```

如果 `DATA_PATH` 不存在，先准备或指定已有数据文件，再进入训练步骤。

## 早期 text 数据格式与转换（历史）

以下保留最初运行使用的 `text` 数据路径；当前示例仍支持它，但不适用于
按 prompt 分组或 completion-only 监督。示例通过 `datasets.load_dataset("json", ...)`
读取 JSON/JSONL，`text` 模式默认要求列名为 `text`。每一行应是已经使用目标模型
chat template 格式化好的完整训练文本，例如：

```json
{"text": "<已经由 Gemma-4 tokenizer chat template 生成的完整对话文本>"}
```

原始 `instruction`、`input`、`output` 或 `messages` 列不会由当前 Demo 自动
转换。需要先使用 Gemma-4 tokenizer 的 `apply_chat_template()` 生成文本，
再写入 `text` 列。

### 数据转换（OIG → Gemma-4 格式）

`unified_chip2.jsonl` 已包含 `text` 列，但格式是 OIG 的 `<human>: ... <bot>: ...`，
不是 Gemma-4 的 chat template 格式。需要先转换。下面的脚本将 OIG 格式的
`<human>: ... <bot>: ...` 解析为 messages 结构，再用 Gemma-4 tokenizer
重新格式化。此次远端只读审计确认原文件共 210,289 行，全部为单轮
`human → bot`，没有格式异常。解析器仍检查全部角色标记并支持多轮；遇到
缺失角色、空内容、额外前缀或未解析文本会报出行号，不跳过异常记录。
正文只去掉每条消息首尾的分隔空白。输出先写同目录临时文件，全部成功后再
原子替换目标文件；中途失败不会留下部分转换结果：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import re
import tempfile
from collections import Counter

from transformers import AutoTokenizer

model_path = Path("/mnt/data8t/models/google--gemma-4-E2B-it/snapshots/master/")
input_path = Path("/workspace/data/unified_chip2.jsonl")
output_path = Path("/workspace/data/gemma4_chip2.jsonl")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

oig_pattern = re.compile(
    r"<human>:\s*(.*?)\s*<bot>:\s*(.*?)(?=\s*<human>:|\s*$)",
    re.DOTALL,
)

def to_messages(row):
    text = row.get("text") if isinstance(row, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("row needs a non-empty text field")
    messages = []
    pos = 0
    for m in oig_pattern.finditer(text):
        if text[pos:m.start()].strip():
            raise ValueError("unparsed text before a conversation turn")
        user_text = m.group(1).strip()
        bot_text = m.group(2).strip()
        if not user_text or not bot_text:
            raise ValueError("empty user or assistant content")
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": bot_text})
        pos = m.end()
    markers = re.findall(r"<(human|bot)>:", text)
    if (not messages or text[pos:].strip()
            or markers != ["human", "bot"] * (len(messages) // 2)):
        raise ValueError("unparsed text or invalid OIG role sequence")
    return messages

row_count = 0
role_counts = Counter()
turn_counts = Counter()
temporary_path = None
try:
    with input_path.open(encoding="utf-8") as source, tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent,
        prefix=output_path.name + ".", suffix=".tmp", delete=False,
    ) as target:
        temporary_path = Path(target.name)
        for line_number, line in enumerate(source, 1):
            try:
                messages = to_messages(json.loads(line))
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("chat template produced empty text")
                # Check that every message body survives formatting, in order.
                position = 0
                for message in messages:
                    found = text.find(message["content"], position)
                    if found < 0:
                        raise ValueError("chat template lost message content")
                    position = found + len(message["content"])
            except Exception as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            target.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            row_count += 1
            role_counts.update(message["role"] for message in messages)
            turn_counts[len(messages) // 2] += 1
        if row_count == 0:
            raise ValueError("input file has no rows")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary_path, output_path)
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)

print(f"Converted {row_count} rows -> {output_path}")
print("roles:", dict(role_counts), "turns per row:", dict(turn_counts))
PY
```

当前数据预期输出 210,289 行，user 和 assistant 各 210,289 条。转换后检查
完整行数和所有文本非空，而不是只检查前几个样本：

```bash
python - <<'PY'
from datasets import load_dataset

path = "/workspace/data/gemma4_chip2.jsonl"
dataset = load_dataset("json", data_files={"train": path}, split="train")
print(dataset)
print("columns:", dataset.column_names)
assert "text" in dataset.column_names
assert len(dataset) == 210289
assert all(isinstance(item, str) and item.strip() for item in dataset["text"])
print("Sample text[:300]:", repr(dataset["text"][0][:300]))
PY
```

## 先做 smoke test

先用短序列、少量样本和 20 step 验证模型加载、目标层匹配、反向传播、保存
adapter 的链路。默认开启 gradient checkpointing，以降低显存压力：

```bash
python /workspace/nanoft/examples/lora_sft_demo.py train \
  --model-name "$MODEL_PATH" \
  --data-path "/workspace/data/gemma4_chip2.jsonl" \
  --text-column text \
  --target-modules "$TARGET_MODULES" \
  --output-root "$OUTPUT_ROOT/smoke" \
  --dtype auto \
  --max-seq-length 512 \
  --max-samples 64 \
  --max-steps 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --save-steps 20
```

smoke test 成功后应存在：

```text
$OUTPUT_ROOT/smoke/
├── trainer/
├── adapter_native/
├── adapter_peft/
└── run_metrics.json
```

检查 `run_metrics.json` 中的 `device`、`model_dtype`、`target_modules` 和
`train_metrics`，并确认日志没有 CUDA out of memory、NaN 或 Inf。smoke 输出
只用于链路验证，不作为正式模型结果。

## 正式训练

smoke test 通过后，先以一个较短但有代表性的步数运行正式输出目录。下面的
`512` step 是起始建议，应根据数据规模、held-out loss 和生成质量调整：

```bash
python /workspace/nanoft/examples/lora_sft_demo.py train \
  --model-name "$MODEL_PATH" \
  --data-path "/workspace/data/gemma4_chip2.jsonl" \
  --text-column text \
  --target-modules "$TARGET_MODULES" \
  --output-root "$OUTPUT_ROOT/run" \
  --dtype auto \
  --max-seq-length 2048 \
  --max-samples 2048 \
  --max-steps 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 10 \
  --learning-rate 2e-4 \
  --logging-steps 1 \
  --save-steps 100
```

训练结束后，正式输出目录为：

```text
$OUTPUT_ROOT/run/
├── trainer/                 # TRL checkpoint
├── adapter_native/          # NanoFT 原生 adapter
├── adapter_peft/            # PEFT 兼容 adapter
└── run_metrics.json         # 运行参数和训练指标
```

`--save-merged` 可以在训练结束时同时导出完整模型，但会额外占用磁盘和内存。
建议先保存 adapter，完成评估后单独运行下一节的 merge。

## 评估 adapter

Demo 的默认生成提示是 OIG `<human>/<bot>` 格式。评估 Gemma 时，先在同一
shell 中由本地 tokenizer 生成带 `{prompt}` 占位符的模板，再传给
`--prompt-template`，使生成时的输入格式与训练格式一致。命令替换会删除末尾
换行，因此先附加固定尾标记，再由 shell 去掉标记，保留模板的全部换行：

```bash
PROMPT_TEMPLATE="$(python - <<'PY'
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True)
marker = "NANOFT_PROMPT_PLACEHOLDER"
template = tokenizer.apply_chat_template(
    [{"role": "user", "content": marker}],
    tokenize=False, add_generation_prompt=True,
)
assert template.count(marker) == 1
template = template.replace("{", "{{").replace("}", "}}").replace(marker, "{prompt}")
print(template + "__NANOFT_PROMPT_END__", end="")
PY
)"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE%__NANOFT_PROMPT_END__}"
export PROMPT_TEMPLATE
test -n "$PROMPT_TEMPLATE"
```

先用 base 与 native adapter 做评估。`eval-offset` 应指向训练数据之后的
held-out 区域；如果数据集少于 2048 行，应显式改成可用的偏移量：

```bash
python /workspace/nanoft/examples/lora_sft_demo.py evaluate \
  --model-name "$MODEL_PATH" \
  --data-path "/workspace/data/gemma4_chip2.jsonl" \
  --text-column text \
  --output-root "$OUTPUT_ROOT/run" \
  --dtype auto \
  --prompt-template "$PROMPT_TEMPLATE" \
  --variants base,native \
  --eval-offset 2048 \
  --eval-samples 64 \
  --eval-max-length 1024 \
  --max-new-tokens 256
```

该命令会写入 `$OUTPUT_ROOT/run/evaluation_results.json`，其中包含 held-out
loss、perplexity 和确定性生成。若没有 2048 行可供留出，应改用例如
`--eval-offset 0 --eval-samples 16`，并在验收记录中注明这不是独立 held-out
切分。需要评估 PEFT adapter 时：

```bash
python /workspace/nanoft/examples/lora_sft_demo.py evaluate \
  --model-name "$MODEL_PATH" \
  --data-path "/workspace/data/gemma4_chip2.jsonl" \
  --text-column text \
  --output-root "$OUTPUT_ROOT/run" \
  --dtype auto \
  --prompt-template "$PROMPT_TEMPLATE" \
  --variants base,native,peft \
  --eval-offset 2048 \
  --eval-samples 64
```

## 合并并验证完整模型

确认 adapter 结果可接受后，运行 merge。输出目录必须不同于 adapter 目录：

```bash
python /workspace/nanoft/examples/lora_sft_demo.py merge \
  --model-name "$MODEL_PATH" \
  --adapter "$OUTPUT_ROOT/run/adapter_native" \
  --output-dir "$OUTPUT_ROOT/run/merged_model" \
  --dtype auto \
  --verify
```

成功时应看到 `Standard Transformers model written to ...` 和
`Verification passed: reloaded model matches in-memory merged model`。该命令
会重新加载导出的完整模型并比较 logits，随后可以评估 merged 版本：

```bash
python /workspace/nanoft/examples/lora_sft_demo.py evaluate \
  --model-name "$MODEL_PATH" \
  --data-path "/workspace/data/gemma4_chip2.jsonl" \
  --text-column text \
  --output-root "$OUTPUT_ROOT/run" \
  --dtype auto \
  --prompt-template "$PROMPT_TEMPLATE" \
  --variants base,native,peft,merged \
  --eval-offset 2048 \
  --eval-samples 64
```

## 常见故障

- **`ModuleNotFoundError: trl` 或 `ModuleNotFoundError: peft`**：确认在
  容器内执行 `cd /workspace/nanoft && pip install -e '.[trl,peft]'`；安装后
  重新做导入检查。
- **`bitsandbytes` 缺失或 CUDA backend 不可用**：本文普通 LoRA 不要求调用
  bitsandbytes。不要因为标准 SFT 失败就切换到 QLoRA。
- **`Expected a non-empty string`**：检查 `--data-format` 对应的列。
  `text` 模式要求 `--text-column` 指向非空字符串；`prompt-completion`
  模式要求两个非空、已经格式化的字符串列。旧 `text` 转换输出不能直接
  用于 completion-only 训练或 prompt 分组。
- **`No nn.Linear modules matched target_modules`**：确认加载的是
  `google/gemma-4-E2B-it`，并检查目标层名称。默认的
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 在本次模型结构
  检查中匹配 205 个文本 Linear 层；模型或配置更换后应重新检查。
- **CUDA out of memory**：RTX 5090 有 32 GB 显存，Gemma-4 5.1B 在 bf16 下
  约占用 10 GB。如果 OOM，先降低 `--max-seq-length`、`--batch-size` 或使用
  更短的序列；保持 `--gradient-checkpointing` 开启。BF16 和 FP16 都是每元素
  2 字节，切换到 FP16 不会减少权重显存；减少 step 数也不会降低单步显存。
- **CUDA 不可用或设备名称不对**：重新运行 `nvidia-smi` 和 PyTorch 检查，
  确认容器启动时传入了 `--gpus all`。
- **TRL/Transformers 参数或 API 不兼容**：记录 `torch`、`transformers`、
  `trl`、`accelerate` 和 `peft` 版本；确认使用的是项目当前
  `pyproject.toml` 的 extras。
- **merge 输出目录已存在或与 adapter 相同**：使用新的
  `$OUTPUT_ROOT/run/merged_model`，不要把 `--output-dir` 指向
  `adapter_native`。
- **模型是多模态的**：Gemma-4-E2B-it 是一个多模态模型（文本+图像+音频），
  本次 Transformers 5.17 对完整 `Gemma4Config` 的 `AutoModelForCausalLM`
  映射为 `Gemma4ForConditionalGeneration`。模型结构检查显示上述七个目标
  后缀只匹配 `model.language_model` 下的 205 个 Linear 层，没有匹配 vision
  或 audio 层。完整模型保留其他子模块；本流程仅验证文本训练和生成，没有
  验证训练后的图像或音频效果。

## 验收标准

一次远端运行至少应满足：

1. 环境检查确认当前 Python 来自容器，CUDA 可用，GPU 为 RTX 5090（32 GB）；
   `trl`、`peft` 能导入。
2. 数据字段与所选格式一致，使用目标模型的 chat template；新分组数据应
   检查 manifest、prompt 精确交集和实际 labels，确认截断后仍有监督 token。
3. smoke test 以退出码 0 完成，生成 native/PEFT adapter 和
   `run_metrics.json`；日志、指标和 adapter 权重没有 NaN/Inf。
4. 正式训练生成 `trainer/`、`adapter_native/`、`adapter_peft/` 和
   `run_metrics.json`，训练指标可解析且保存过程无错误。
5. `evaluate` 生成 `evaluation_results.json`，记录 variants、数据范围、
   监督口径、loss/perplexity、生成和实际执行配置；核对各阶段 AMP、dtype、
   attention 与确定性设置。单独记录任务正确性与指令遵循，披露数据重合限制。
6. `merge --verify` 成功，显示重新加载后的模型与内存中 merged model 一致，
   并在 `merged_model/` 中写出可由 Transformers 重新加载的完整模型。

## 2026-09-20 实测记录

在 `203` 的 `zrk_worker` 容器、RTX 5090 上验证。开始时宿主 GPU 正常、
容器 NVML 异常；确认容器空闲后重启，CUDA 与 BF16 运算恢复。运行版本为
PyTorch `2.8.0+cu128`、Transformers `5.17.0`、TRL `1.13.0`、PEFT `0.21.0`、
Accelerate `1.15.0`、Datasets `5.0.1`。训练启动时，本地与远端的 11 个源码及
项目配置文件 SHA256 一致，远端测试为 `74 passed in 3.43s`。随后修复下述
PEFT 导出问题并同步源码与回归测试，当时的测试为 `75 passed in 3.39s`。

OIG 原始数据全部 210,289 行转换成功，每条消息的角色和正文均校验保留，
输出文本均非空。提取本文最终转换脚本在远端另行全量重跑，输出逐字节一致：

```text
文件：/workspace/data/gemma4_chip2.jsonl
大小：93,557,968 bytes
SHA256：2e13519522a55e67b40f011336c1edc1a9399372ab30b9292427e320467e9a97
```

训练使用前 2048 行；评估使用从 offset 2048 开始的 64 行，两个切分之间
没有完全相同的文本。

| 阶段 | 实际配置 | 已验证结果 |
| --- | --- | --- |
| Smoke | 64 样本，20 step，长度 512，batch 1，accumulation 1 | 退出码 0，train loss 3.787879 |
| 正式训练 | 2048 样本，512 step，长度 2048，batch 1，accumulation 4，学习率 2e-4，LoRA r/alpha 16/16，BF16 | 退出码 0，epoch 1.0，train loss 1.5045280862832442 |

正式训练计时为 398.5805 秒，包含模型加载等准备阶段的总耗时为 420.0189 秒。
512 step 的数值指标全部有限；native 和 PEFT adapter 各含 410 个有限值
tensor，205 个 LoRA B 矩阵均有更新。

首次 PEFT 评估加载失败：按后缀导出的 `target_modules` 还会让 PEFT 匹配
Gemma 的视觉包装层 `Gemma4ClippableLinear`。NanoFT 的训练匹配限定为
`nn.Linear`，实际训练的 205 个文本层没有受到影响。修复后，PEFT 导出配置
只列出实际适配的 205 个完整模块名；native 导出范围不变。无需重训，重新
导出的 410 个权重 tensor 逐一相同，safetensors 文件的 SHA256 也保持为
`08f65347a4346ec51f38ffe6283e42ad397bf6eadf99078adb1827050d1310da`。
旧导出保留在远端 `run/adapter_peft_initial/`，原失败日志及退出码保留在
`evaluate.initial.log`、`evaluate.initial.exit`，后者为 1。

最终四种模型使用同一 64 条评估文本，共 6359 个计分 token：

| 模型 | Loss | Perplexity |
| --- | ---: | ---: |
| Base | 4.860554 | 129.095764 |
| Native adapter | 1.398722 | 4.050019 |
| PEFT adapter | 1.398213 | 4.047961 |
| Merged | 1.397704 | 4.045901 |

BF16 合并与保存重载是两项独立比较：

| 比较 | 最大绝对 logits 差 | 平均绝对 logits 差 | Top-1 一致率 |
| --- | ---: | ---: | ---: |
| Native adapter → 内存中 merged | 1.34375 | 0.160683 | 94.12% |
| 内存中 merged → 保存后重新加载 | 0 | 0 | 100% |

合并产物包含 1951 个 tensor，没有 LoRA 参数残留。3 条确定性生成用例中，
native 与 merged 的输出文本一致；这不代表两者 logits 完全相等。最终
smoke、train、evaluate、merge、evaluate_merged、post_training 六阶段退出码
均为 0。

本次产物保存在容器 `/workspace/outputs/gemma4-sft-20260920`，对应宿主机
`/home/runke/workspace/outputs/gemma4-sft-20260920`。本地日志归档位于
项目内 `output/gemma4-sft-20260920/`，已包含 `environment.json`、
`data_conversion.json` 和以下复现记录，路径均相对于该输出根目录：

| 内容 | 文件 |
| --- | --- |
| 实际训练命令 | `run_smoke.sh`、`run_train.sh` |
| 后处理命令 | `run_post_training.sh`、`resume_post_training.sh`、`run_merge.sh`、`run_evaluate_merged.sh` |
| 实际评估/合并参数 | `evaluate_command.json`、`merge_command.json`、`evaluate_merged_command.json`、`prompt_template.txt` |
| 成功日志与退出码 | `smoke.log`、`train.log`、`evaluate.log`、`merge.log`、`evaluate_merged.log`、`post_training.log` 及各自的 `.exit` |
| 首次失败记录 | `evaluate.initial.log`、`post_training.initial.log` 及各自的 `.exit` |
| 训练指标 | `smoke/run_metrics.json`、`run/run_metrics.json` |
| 四种模型评估汇总 | `run/evaluation_all_variants.json` |
| 权重与最终校验 | `adapter_validation.json`、`peft_reexport_validation.json`、`final_validation.json`、`pytest_fixed.log` |

这些结果验证了本次小样本文本训练、adapter 互操作、评估和合并导出流程。
Loss 的下降仅适用于该评估切分，不保证任务泛化能力；图像与音频能力未测试。

## 全量数据训练（2026-09-20）

本轮使用全部去重后的数据划分，方法仍为 NanoFT LoRA，仅训练适配器，
不是全参数微调。从原始 base model 重新开始，不继承上述 2048 条小样本
运行的 adapter。前一节的 loss、生成与合并结果属于旧运行，不能作为本轮结果。

全量作业在 `203` 的 `zrk_worker` 容器中于北京时间
**2026-09-20 09:46:31** 启动（UTC `2026-09-20T01:46:31.675175+00:00`），
并于北京时间 **11:46:09** 完成训练及后处理。最终状态为 `completed`，
`pipeline.exit=0`；训练 208234 条、验证 2048 条，完成 **13015 step / 1 epoch**。
启动时记录的驱动 PID `6683`、训练 PID `6686` 仅是历史观测，不代表当前进程。

随后北京时间 09:48:31 的启动检查已到 `103/13015` step；第 100 步训练
loss 为 `1.664`，显存占用 `13458 MiB`，检查到的 5 条日志数值均有限。该观察记录在
`startup_observation.json`，属于启动检查，不是最终训练或验证结果。

本轮输出根目录为容器内
`/mnt/data8t/nanoft-runs/gemma4-full-20260920`，下文记作 `full_root`。
运行入口 `run_full.sh` 和 `run_full.py` 持久保存在该目录；shell 设置运行环境
后执行 Python driver，driver 调用项目现有的
`/workspace/nanoft/examples/lora_sft_demo.py`，训练循环由 TRL 负责。

### 数据、配置与启动前验证

转换后的 210289 条数据按完整文本去除 7 条重复项，保留 210282 条。
使用固定随机种子 `3407` 划分为 `data/train.jsonl` 的 **208234** 条训练数据
与 `data/validation.jsonl` 的 **2048** 条验证数据；两组没有相同完整文本。
`split_manifest.json` 记录切分、源行号及文件 SHA256，入口启动时校验数据
行数和 SHA256。验证行不参与训练，但后续审计发现 **1096/2048 行
（53.515625%）的问题已出现在训练集中**，只是答案不同，因此不能把该划分
称为问题独立的 heldout。当前分组工具用于避免这种精确 prompt 交叉。

| 参数 | 本轮值 |
| --- | --- |
| Base model | `/mnt/data8t/models/google--gemma-4-E2B-it/snapshots/master/` |
| 训练轮数 | 1 epoch，实际 `ceil(208234 / 16) = 13015` 个 optimizer step |
| Batch / gradient accumulation | 16 / 1，单 GPU |
| 精度 / 最大序列长度 | BF16 / 2048 |
| LoRA r / alpha | 16 / 16 |
| 学习率 / warmup | `2e-4` / 391 step |
| 日志 / 验证 / checkpoint 间隔 | 20 / 1000 / 1000 step |
| Checkpoint 保留 | 最近 2 份，位于 `run/trainer/` |

训练参数 `--max-samples 0` 使用全部训练文件，`--max-steps -1` 使
`--num-train-epochs 1` 控制训练长度；`--eval-data-path` 指向独立验证文件，
`--eval-steps 1000` 开启周期 loss 验证。环境固定为 `CUDA_VISIBLE_DEVICES=0`、
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONUNBUFFERED=1`、
`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`；`HF_DATASETS_CACHE` 指向
`$full_root/datasets_cache`，缓存和产物使用大盘。

启动前，远端 CLI 帮助与完整测试通过：`75 passed in 3.30s`。真实恢复短测
从 checkpoint 20 继续到 22，只新增 step 21、22；optimizer 状态到 step 22，
410 个 adapter tensor 均有更新且全部有限，step 22 的 `eval_loss` 为
`2.0004782676696777`。这些结果验证恢复和周期验证路径，不代表全量训练效果。
证据见 `preflight_cli_tests.log`、`launcher_validation_remote.log` 和
`resume_validation.json`。

### 启动、状态查看与恢复

以下是本轮历史启动命令，在 `zrk_worker` 容器内执行。该运行已经完成，
保留命令用于追溯，不应重新启动已完成的目录。入口通过文件锁防止重复任务，
首次启动遇到非空 `run/` 会拒绝覆盖：

```bash
full_root=/mnt/data8t/nanoft-runs/gemma4-full-20260920
nohup bash "$full_root/run_full.sh" \
  >> "$full_root/launcher.log" 2>&1 < /dev/null &
```

从本机通过 `ssh 203` 查看持久状态和最新训练日志：

```bash
ssh 203 'docker exec zrk_worker cat /mnt/data8t/nanoft-runs/gemma4-full-20260920/pipeline_status.json'
ssh 203 'docker exec zrk_worker tail -n 40 /mnt/data8t/nanoft-runs/gemma4-full-20260920/train.log'
```

中断后，选择本轮相同配置下实际存在的 checkpoint，在容器内显式恢复。
例如已有 `run/trainer/checkpoint-1000` 时：

```bash
full_root=/mnt/data8t/nanoft-runs/gemma4-full-20260920
checkpoint="$full_root/run/trainer/checkpoint-1000"
test -f "$checkpoint/trainer_state.json"
nohup bash "$full_root/run_full.sh" --resume-from-checkpoint "$checkpoint" \
  >> "$full_root/launcher.log" 2>&1 < /dev/null &
```

`--resume-from-checkpoint` 透传给 TRL，恢复训练权重与 Trainer 状态；没有自动
重试。各阶段日志追加保留，训练退出非零时停止后续处理。

### 后处理与完成判据

训练成功后，driver 校验 `run/run_metrics.json` 中的样本数、实际 global step、
epoch 和数值有限性，再依次执行 base/native/PEFT 评估、`merge --verify`、
merged 评估。四种模型都使用独立验证文件的全部 2048 条文本，offset 为 0、
最大长度为 2048；生成最多 256 个新 token，提示模板从 `prompt_template.txt`
读取。最终指标合并为 `run/evaluation_all_variants.json`，检查所有数值有限。

| 记录 | 文件（相对于 `full_root`） |
| --- | --- |
| 当前阶段、状态、PID、时间 | `pipeline_status.json`，原子更新 |
| 实际命令和历史 | `train_command.json`、`evaluate_command.json`、`merge_command.json`、`evaluate_merged_command.json`、`commands.jsonl` |
| 完整阶段日志和退出码 | `train.log`、`evaluate.log`、`merge.log`、`evaluate_merged.log` 及各自的 `.exit` |
| 流程日志和最终退出码 | `pipeline.log`、`pipeline.exit` |
| 训练产物和验收 | `run/adapter_native/`、`run/adapter_peft/`、`run/run_metrics.json`、`training_validation.json` |
| 合并模型和评估汇总 | `run/merged_model/`、`run/evaluation_all_variants.json` |

仅当 `pipeline.exit` 为 `0` 且 `pipeline_status.json` 的 `status` 为
`completed` 时，整条流水线才算完成；本轮已满足该流程判据。

### 全量最终结果及其限制

独立评估使用同一 2048 行、189742 个计分 token：base 的 all-text loss 为
`4.9869420209`，Native adapter 为 `1.1622481032`。Native、PEFT 和 merged
均完成评估；merged 保存重载与内存中合并模型的检查提示 logits 差为 0。
该检查不保证合并前后所有输入逐位相同，BF16 合并仍存在舍入差异。

问题重合和训练分布使上述 loss 不能代表泛化提升。后续实际问题评估观察到
CSV 指令转成 Python 程序、重复输出至长度上限等回归，因此本轮是流程完成，
不是综合效果验收通过。终态和指标可在 203 容器内复核：

```text
/mnt/data8t/nanoft-runs/gemma4-full-20260920/pipeline_status.json
/mnt/data8t/nanoft-runs/gemma4-full-20260920/training_validation.json
/mnt/data8t/nanoft-runs/gemma4-full-20260920/run/evaluation_all_variants.json
/mnt/data8t/nanoft-runs/gemma4-full-20260920/question-eval-20260920/
```

## 问题分组与精度受控短训（2026-09-20）

后续实验使用 4096 条训练、256 条验证（225 个不同问题），按完整问题分组，
两侧精确交集为零。五臂各训练 256 step：全 token / assistant-only 监督与
BF16 / FP32 A/B 的四个 NanoFT 组合，加一个 assistant-only + FP32 PEFT
参照。基座、初始化、数据顺序、优化器、eager attention 和 BF16 AMP 保持一致。

同条件 Native/PEFT 的 **410 个矩阵、24,158,208 个元素**在最终 checkpoint
逐位相同，30 条诊断提示的生成 token 也全部相同。它支持本次文本训练路径的
实现一致性，不能推广为所有模型、硬件或多模态训练都等价。

问题诊断共 6 组模型 × 30 条提示：指令题 base 通过 5/6，五个短训模型均为
2/6；常规 Python base 通过 4/6，短训模型为 1–3/6。中文冻结规则有部分改善，
不能说所有能力都下降；FP32 与 assistant-only 也没有自动解决效果回归。
题目包含已知失败提示及相同任务的变体，是小样本诊断，不是未见泛化 benchmark。

训练回调与 fresh 评估曾因执行配置组合不一致而未通过 loss 精确相等断言，
原 `generation_validation.exit=1` 和日志保留。探针对 3 个模型各 8 条样本，
分别复现训练配置与独立推理配置下的原 NLL；差异涉及 CUBLAS workspace、
CUDA blocking、cuDNN deterministic 等组合，不能指定其中一个变量为唯一
原因，也不能把原完整 256 行断言改判通过。当前 CLI 在三个阶段共同设置并
记录执行环境，以便后续在相同条件下比较。

本轮没有证据确认基座预训练见过 OIG，也没有证明经典过拟合或某个数据尾句
是唯一原因。后续优先清理与补充目标任务数据，在固定任务评价通过后再扩大
训练；精确 prompt 隔离不排除语义重合。203 容器中的原始产物根目录为：

```text
/mnt/data8t/nanoft-runs/gemma4-controlled-20260920/
  arms/<arm>/checkpoint-{128,256}/adapter/
  training_summary.json
  generation_integrity.json
  runner/generation_validation.log
  runner/generation_validation.exit
  execution_context_probe/
```

这些验证均为文本 LoRA。真实图像/音频训练及 CUDA QLoRA 的反向传播、保存、
重载闭环尚未验收。

## 正式入口收口验收（2026-09-20）

同一源码快照的完整 CPU 套件为 **123 passed、2 个 CUDA 用例 skipped**，
随后 CUDA 专项 **2 passed**，共覆盖 125 个用例。CPU 集成实际验证了
completion labels 进入 TRL dataloader、格式化生成只有一个 BOS、按一轮
训练，以及从 checkpoint 2 恢复到 step 3 和 optimizer 状态恢复。

新入口还在上述 203/Gemma 环境执行了 **2 条样本、2 step** smoke：train、
evaluate、merge、evaluate_merged 四阶段均 exit 0。Native/PEFT 各 410 个
FP32 adapter tensor 重载后分别与保存权重逐位相同，二者本次 loss 和生成相同。各阶段实际
AMP、确定性、TF32、CUDA 环境和 eager attention 已核对一致。

本次检查提示上，adapter → BF16 merged 的 logits 最大绝对差为 `0.875`、
平均差为 `0.230264`；merged → 保存重载差为 `0`。两项比较独立记录，
不能把重载无损写成合并前后数值相同。原始命令、阶段日志、退出码和
`validation.json` 位于 203 容器：

```text
/mnt/data8t/nanoft-runs/sft-entry-consolidation-20260920/entry/gemma-smoke/
```

这次 smoke 仅用于入口与产物流程验收，没有重新评价未见问题或证明任务质量
提升；前述完整训练与受控短训的回归结论保持不变。
