# NanoFT Engineering Notes

NanoFT should stay lightweight and focused on fine-tuning primitives. Treat it as
a fine-tuning framework and toolkit that prepares models, adapters, data, and
integration surfaces for training frameworks, not as a training-loop framework.

## Project Boundary

NanoFT owns:

- Fine-tuning method implementations, such as LoRA layers, adapter injection,
  parameter freezing, adapter merge/unload, and adapter save/load.
- Data preparation utilities for fine-tuning workflows, including prompt/chat
  formatting, dataset normalization, tokenization helpers, and format
  conversion.
- Adapter and model I/O, including NanoFT-native adapter format and explicit
  compatibility exports for external ecosystems.
- Integration helpers, migration docs, and basic examples that make NanoFT
  outputs easy to use with TRL, Transformers, PyTorch, Accelerate, or similar
  training frameworks.

NanoFT does not own:

- Full training loops.
- Distributed training orchestration.
- Optimizer, scheduler, checkpoint, logging, or experiment-management stacks.
- Reimplementations of `torch` training code, `transformers.Trainer`,
  `trl.SFTTrainer`, `trl.DPOTrainer`, or equivalent framework trainers.

When adding examples, keep training execution delegated to external frameworks
and make that delegation explicit. Examples may show how to pass a NanoFT-patched
model or NanoFT-prepared dataset into `trl`, `torch`, or `transformers`, but the
core package should not grow a competing trainer abstraction. Framework
integration examples are part of the project surface and should be maintained
for common migration paths.

## Design Principle

Prefer small, composable APIs over framework-level orchestration. If a feature is
primarily about how to run optimization steps, schedule jobs, scale across
devices, or manage long-running training state, it belongs in TRL, PyTorch,
Transformers, Accelerate, or user code. If a feature is about preparing model
structure, adapter weights, or data formats so those tools can train correctly,
it can belong in NanoFT.
