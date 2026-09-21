# Merge 导出校验失败

## 现象

运行：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/lora_sft_demo.py merge
```

日志显示模型已写出，随后 verification 报错：

```text
Standard Transformers model written to outputs/qwen3_0_6b_lora_sft-max/merged_model
Reloading exported model for verification
RuntimeError: Reloaded merged model does not match adapter model:
  max_abs_diff=0.875, tolerance=0.02
```

## 结论（先看这个）

| 项目 | 结论 |
|------|------|
| 合并是否成功 | **成功**。`merged_model/` 在报错前已写出 |
| 失败环节 | **导出后的 logits 校验**，不是 `save_merged_model` 本身 |
| 根因 | 用「未 merge 的 LoRA 前向」对比「已 merge 的 Linear」；bf16 下数值不等价 |
| 修复 | 改为用内存中已 merge 的模型做 expected，再与 reload 结果对比 |

## 根因分析

### 1. 脚本原先的校验顺序

旧逻辑大致是：

1. `load_adapter` 加载 LoRA
2. `model.eval()`
3. **立刻**对当前模型算 `expected_logits`（此时仍是 LoRA 分支前向）
4. `save_merged_model(..., inplace=True)` 合并并落盘
5. 重新 `from_pretrained(merged_model)` 算 `actual_logits`
6. `allclose(expected, actual, atol=0.02)`

第 3 步和第 5 步走的不是同一条计算路径。

### 2. NanoFT 不会在 `eval()` 时自动 merge

训练/加载后的 `LoRALinear` 使用 `merge_weights=False`。
因此 `model.eval()` **不会**把 `B @ A * scaling` 写进 base weight。

实测（`adapter_native`）：

```text
n_lora=196, merge_weights=False, merged=False
after eval → merged 仍为 False
```

于是 `expected_logits` 实际走的是：

```text
y = W·x + (x @ Aᵀ @ Bᵀ) * scaling     # 未 merge 路径
```

而导出后的模型走的是：

```text
W' = W + (B @ A) * scaling
y  = W'·x                              # 已 merge 路径
```

数学上在 float32 下应接近；在 **bfloat16** 下，矩阵乘累加顺序不同，logits 会出现明显偏差。

### 3. 误差量级与阈值

| 项 | 值 |
|----|----|
| 实测 `max_abs_diff` | `0.875` |
| 脚本 bf16 容差 | `0.02` |
| 判定 | 失败（误报） |

这是 **数值路径不一致**，不是权重没合并或文件损坏。

## 修复方式

`examples/lora_sft_demo.py merge` 已改为：

1. merge 前保留 adapter logits，只用于报告低精度 merge drift
2. 调用 `save_merged_model`（内存模型变成普通 `nn.Linear`）
3. 用**内存中已 merge 的模型**算 `merged_logits`
4. reload 落盘模型算 `reloaded_logits`
5. 用 `merged_logits` 对比 `reloaded_logits`，校验保存/加载是否忠实

修复后重跑结果：

```text
Adapter -> in-memory merged drift: max_abs_diff=0.875, ...
Verification passed: reloaded model matches in-memory merged model;
  max_abs_diff=0, mean_abs_diff=0, top1_agreement=100.00%
```

## 临时绕过（未打补丁时）

若只想导出、不跑校验：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/lora_sft_demo.py merge --no-verify
```

此前已生成的 `outputs/qwen3_0_6b_lora_sft-max/merged_model/` 一般可直接使用；需要严格确认时，用修复后的脚本再导出一遍。

## 相关文件

- 脚本：`examples/lora_sft_demo.py merge`
- Adapter：`outputs/qwen3_0_6b_lora_sft-max/adapter_native/`
- 合并产物：`outputs/qwen3_0_6b_lora_sft-max/merged_model/`
- NanoFT 行为：`LoRALinear.train()/eval()` 仅在 `merge_weights=True` 时自动 merge；本流程为 `False`

## 经验小结

对 LoRA merge 做数值校验时：

1. 可以记录「unmerged LoRA forward」与「merged Linear」的差异，但在
   bf16/fp16 下只能把它当作 merge drift 指标，不能用于判定序列化失败
2. 导出校验应对齐同一语义：例如「内存 merge 结果」vs「磁盘 reload 结果」
3. 日志里若已出现 `Standard Transformers model written to ...`，优先检查校验逻辑，而不是先怀疑权重没写出
