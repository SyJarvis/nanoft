# Roadmap

NanoFT 的长期定位是一个轻量、可组合、和 HuggingFace 生态友好的
LLM post-training primitives 工具包。它负责准备模型、adapter 和数据，
让 SFT、Preference Optimization 和 RL 训练框架可以直接消费这些标准对象；
它不替代 TRL、Transformers、PyTorch、Accelerate 或专门的 RL 训练系统。

## 总路线

NanoFT 参考 Unsloth 的方向，但不复制它的边界。Unsloth 值得借鉴的是低门槛入口、常见模型的快速适配、LoRA/QLoRA 的易用封装、adapter 保存与合并体验，以及和 TRL 训练脚本的顺滑组合。NanoFT 要保留这些“好用”的部分，同时保持更小的核心：

- **轻量核心**：只维护必要的 LoRA/QLoRA、adapter、merge、load/save、device、data/format 和模型准备工具。
- **框架适配**：训练循环、分布式、optimizer、scheduler、checkpoint、logging、实验管理交给 TRL、Transformers、PyTorch、Accelerate 或用户代码；NanoFT 负责让模型、adapter、数据和配置能顺滑接入这些训练框架。
- **两个正交维度**：LoRA/QLoRA 描述“哪些参数、以何种表示训练”；SFT/DPO/PPO/GRPO 描述“使用什么数据和目标训练”。NanoFT 分别提供参数高效微调 primitives 和任务输入准备，不把两者耦合进 Trainer。
- **格式优先**：把 SFT、Preference、Reward、rollout prompt、chat/instruction 数据格式转成外部训练框架可以直接消费的 dataset、collator 或字段布局。
- **兼容优先**：adapter 默认有 NanoFT 原生格式，同时显式支持 PEFT 兼容导出/导入，合并后的模型保持标准 HuggingFace/PyTorch 可加载。
- **迁移友好**：提供从现有 TRL、Transformers、PyTorch、Accelerate 训练代码迁移到 NanoFT adapter/data 工具的文档和最小案例。
- **端侧友好**：优先保证 CUDA、Apple MPS、CPU 上的可运行性和有限资源下的稳定性，再逐步补充性能优化。
- **可验证输出**：adapter、merged model、转换后的数据都应易检查、可复现、可用外部工具继续处理。

训练框架适配层的 demo 反推设计见
[训练框架集成设计](integration-design.md)。该文档以 Unsloth 对比脚本和
NanoFT + TRL demo 为依据，拆解出 v0.2 需要补齐的模型加载、LoRA 准备、
数据加载和输出保存能力。

## 不走的路线

NanoFT 不发展成训练执行框架：

- 不实现自己的 `Trainer`。
- 不封装完整分布式训练栈。
- 不维护复杂 optimizer/scheduler/checkpoint/logging 体系。
- 不用项目核心代码替代 `trl.SFTTrainer`、`trl.DPOTrainer`、`transformers.Trainer`。
- 不实现 rollout engine、reward 聚合、advantage estimation、KL controller、PPO/GRPO loss 或分布式采样系统。
- 不把 CUDA kernel 或框架级性能魔改作为 v0.x 的主线目标。

但 NanoFT 应该主动适配这些训练框架：维护基础案例、迁移指南、输入输出格式约定和小而薄的 helper，让用户可以把 NanoFT 接入自己的训练代码。

## v0.1 / v0.1.2: 稳定 LoRA/Adapter 核心

v0.1 的目标是把最小可用闭环做稳：注入 LoRA、训练外部执行、保存 adapter、加载 adapter、合并导出标准模型。

已具备或应稳定下来的能力：

- `LoRAConfig`：可序列化配置，兼容 PEFT adapter config 的关键字段。
- `LoRALinear`：替换标准 `nn.Linear`，支持 LoRA 参数训练和权重合并。
- `apply_lora()` / `remove_lora()`：按模块名后缀安全注入和卸载 LoRA。
- `prepare_model_for_training()`：冻结非 LoRA 参数，提供外部 Trainer 可直接接手的模型。
- `save_adapter()` / `load_adapter()`：保存和加载 NanoFT 原生 adapter。
- `save_peft_adapter()`：显式导出标准 PEFT adapter，并用真实 PEFT 做双向互操作测试。
- `merge_and_unload_lora()` / `save_merged_model()`：合并 LoRA 并导出无 LoRA 类依赖的完整模型。
- `detect_device()`：提供 CUDA/MPS/CPU 基础检测。
- 示例脚本：演示如何把 NanoFT 处理后的模型接入 HuggingFace `Trainer`，但不把训练循环纳入 NanoFT。

v0.1 的验收标准：

- 小模型 LoRA 注入、保存、加载、合并流程稳定。
- CUDA 与 MPS 都能完成普通 LoRA 的外部训练、adapter 保存和 merge/export
  闭环；CPU 保留功能回退。
- LoRA forward、梯度、dropout、merge/unmerge 和常用 dtype 有独立数值测试。
- 非法配置、目标层未匹配和重复注入必须明确失败，不能产生零可训练参数或覆盖已有 adapter。
- `none`、`lora_only`、`all` bias 策略在训练、保存和加载阶段行为一致。
- 非有限 adapter 权重和 shape 不匹配必须在保存或加载阶段明确失败。
- 合并后的模型不再包含 LoRA 参数 key，可被标准 Transformers/PyTorch 加载。
- NanoFT adapter 和标准 PEFT adapter 的双向权重互通由集成测试保证。
- README 和示例清楚表达“NanoFT 负责适配训练框架，训练执行由外部框架负责”。

## v0.2: QLoRA 与 SFT 数据准备（进行中）

v0.2 的目标是加入独立、可验证的量化模型准备能力，并补上 SFT
训练前最容易出错的数据格式、chat template、tokenization 和 label mask。
量化计算交给成熟 backend，NanoFT 不实现 NF4 kernel。

当前 0.2.0 已提供 `QuantizationConfig`、`QLoRAConfig`、CUDA +
bitsandbytes 4-bit 包装以及 `dequantize_and_merge()`。这些接口仍属于实验性
能力；真实量化训练需要在 CUDA、bitsandbytes 和外部 Trainer 组合上继续验证。

优先方向：

- **独立配置**：已引入 `QuantizationConfig`，并以
  `QLoRAConfig(lora=LoRAConfig(...), quantization=QuantizationConfig(...))`
  组合 LoRA 与量化配置；普通 `LoRAConfig` 不包含无效量化字段。
- **量化模型准备**：已提供首版 device/backend/dtype 检查、量化 Linear 识别和
  LoRA 注入；embedding、LayerNorm、lm_head 等模块的完整 dtype/冻结策略仍需
  在真实 QLoRA 训练中验证。
- **后端顺序**：首个 QLoRA 闭环以 CUDA 和成熟量化 backend 为验收目标；
  MPS 继续保证普通 LoRA，量化训练等待可验证、可维护的 MPS backend。
- **量化输出语义**：区分 adapter 保存与 `dequantize_and_merge`，不允许普通
  merge API 静默修改量化权重。
- **数据 schema**：定义统一的内部样本结构，覆盖 `messages`、instruction/input/output、prompt/completion 等常见格式。
- **格式转换**：支持 ShareGPT、Alpaca、OpenAI messages、plain prompt/completion 到 SFT 可用格式的转换。
- **chat template 工具**：复用 tokenizer 的 `apply_chat_template`，提供一致的 prompt 构造、截断和 label mask 策略。
- **TRL 集成辅助**：提供小而薄的 helper，输出 `SFTTrainer` 可直接使用的 dataset 字段或 collator 配置，不封装 trainer 本身。
- **训练框架案例**：维护一个模型无关的 `examples/lora_sft_demo.py`，通过 TRL 展示模型注入、外部训练、adapter 保存、评估和合并导出的最小闭环；不为不同模型或 Trainer 复制入口。
- **迁移指南**：说明如何把已有训练脚本迁移到 NanoFT：替换 PEFT/自写 LoRA 注入、接入 NanoFT 数据转换、保存 NanoFT/PEFT adapter、合并导出标准模型。
- **模型目标层预设**：为 Qwen、Llama、Mistral、Gemma 等常见架构提供推荐 `target_modules`，降低手动配置成本。
- **验证工具**：增加 adapter/merged model 检查、有限权重检查、LoRA key 检查、样本格式检查。
- **CLI 小工具**：围绕格式转换、adapter 检查、merge/export 提供命令行入口。

v0.2 的验收标准：

- 用户可以把常见指令微调数据一键转换成 TRL/HF Trainer 友好的结构。
- QLoRA 配置与普通 LoRA 配置相互独立，受支持 backend 的能力和失败模式可测试。
- 量化模型可以完成 adapter 注入、外部 Trainer 反向传播和 adapter 保存闭环。
- 常见模型无需手写大量 target module 名称即可完成 LoRA 注入。
- 数据转换、adapter I/O、merge/export 都有独立测试覆盖。
- 示例覆盖 SFT 数据准备、HuggingFace Trainer/TRL/PyTorch/Accelerate 接入、adapter 合并和推理验证四个环节。

## v0.3: Preference Optimization 准备层

v0.3 优先支持离线偏好优化，不在这一阶段引入在线 rollout 系统。

- 定义 `PreferenceSample`，归一化 `prompt/chosen/rejected` 和 messages 偏好数据。
- 提供 chosen/rejected chat template、tokenization、截断和校验工具。
- 引入命名多 adapter：添加、选择、禁用、冻结以及按名称保存/加载 adapter。
- 准备 policy model 与 reference model/reference adapter，返回标准模型对象。
- 提供 NanoFT + TRL `DPOTrainer` 的基础案例，并逐步验证 IPO、ORPO、KTO、SimPO 所需输入格式。

验收标准：

- 同一 base model 可以安全管理 policy/reference adapter，且只有预期参数可训练。
- 常见 preference 数据可以转换成外部 DPO 类 Trainer 直接消费的结构。
- reference model 或 reference adapter 的冻结、切换和保存行为有独立测试。

## v0.4: 在线 RL 的模型与数据准备

v0.4 面向 PPO、GRPO、RLOO 等在线 RL 框架提供准备 primitives，但训练算法和
rollout 执行仍在 NanoFT 之外。

- 准备 policy、reference、reward model 和可选 value head。
- 定义 rollout prompt 与 reward sample schema。
- 提供模型冻结、adapter 切换、dtype/device 检查和输出保存约定。
- 维护与外部 RL trainer 的最小集成案例。

验收标准：

- 外部 RL trainer 可以直接消费 NanoFT 准备的标准模型、adapter 和数据。
- NanoFT 核心不包含 rollout、advantage、KL、PPO/GRPO loss 或分布式执行代码。

## v0.x 之后

后续版本可以继续扩展，但仍遵守轻量边界：

- 更完整的 PEFT 互操作能力。
- 更多 adapter 类型和参数冻结策略。
- 更好的端侧推理辅助和导出体验。
- 更系统的性能基准与兼容性矩阵。
- 与 ModelScope、HuggingFace Hub 等模型/数据源的轻量集成。

核心判断标准保持不变：如果功能是在“准备模型、adapter 或数据”，可以进入
NanoFT；如果功能是在“定义、执行和管理训练算法”，应交给外部框架。
