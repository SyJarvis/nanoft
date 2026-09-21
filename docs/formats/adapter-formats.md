# NanoFT 与 PEFT Adapter 格式

NanoFT 支持两种显式输出格式。二者使用相同的 LoRA 数值和
`adapter_config.json`，区别主要在 state dict key。

## NanoFT 原生格式

原生 key 与 NanoFT 模型内部 `state_dict()` 一致：

```text
model.layers.0.self_attn.q_proj.lora_A
model.layers.0.self_attn.q_proj.lora_B
```

NanoFT 只保留两个直接参数，不引入 runtime adapter 名称：

```text
lora_A
lora_B
```

因此原生格式中没有 `lora_A.default.weight`，也没有
`lora_A.weight`。

## PEFT 兼容格式

调用 `save_peft_adapter()` 后，key 转换为：

```text
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
```

PEFT 保存文件通常也不包含 runtime adapter 名称 `default`。NanoFT 在
加载时会移除 `base_model.model.` 前缀和 `.weight` 后缀，恢复内部 key。

## 目录结构

两种格式都使用：

```text
adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

也支持 `adapter_model.bin`，但 safetensors 是默认和推荐格式。

## 选择哪一种

| 使用场景 | 推荐格式 |
|---|---|
| NanoFT 加载、继续训练或合并 | NanoFT native |
| 交给 PEFT `PeftModel` | PEFT compatible |
| 不希望运行时依赖 adapter 框架 | merged model |

`load_adapter()` 可以读取 NanoFT 和 PEFT 两种格式，因此无需为了 NanoFT
加载而提前转换。
