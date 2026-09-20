"""One model-agnostic NanoFT LoRA SFT demo.

NanoFT prepares the model and saves adapters. TRL owns the training loop.
Qwen3-0.6B and Unified CHIP2 are runnable defaults, not requirements.

Examples:
    python examples/lora_sft_demo.py train
    python examples/lora_sft_demo.py evaluate --variants base,native
    python examples/lora_sft_demo.py merge
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from nanoft import (
    LoRAConfig,
    LoRALinear,
    detect_device,
    get_recommended_dtype,
    load_adapter,
    prepare_model_for_training,
    print_trainable_parameters,
    save_adapter,
    save_merged_model,
    save_peft_adapter,
)
from transformers import __version__ as transformers_version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    enable_full_determinism,
    set_seed,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_DATA_PATH = (
    "https://huggingface.co/datasets/laion/OIG/resolve/main/"
    "unified_chip2.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/lora_sft_demo")
DEFAULT_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)
DEFAULT_PROMPT_TEMPLATE = "<human>: {prompt}\n<bot>:"
DEFAULT_PROMPTS = [
    "请简单解释 LoRA 的作用。",
    "Write a Python function that returns the Fibonacci sequence.",
    "What are three practical ways to reduce household energy use?",
]
SUPPORTED_VARIANTS = {"base", "native", "peft", "merged"}
DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")
DETERMINISM_ENV = {
    "CUDA_LAUNCH_BLOCKING": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "HCCL_DETERMINISTIC": "1",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
}


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="auto",
        help=(
            "Model dtype. auto uses BF16/FP16 on CUDA and stable FP32 on "
            "MPS/CPU; MPS FP16 can be selected explicitly for lower memory."
        ),
    )
    parser.add_argument(
        "--adapter-dtype", choices=DTYPE_CHOICES, default="auto",
        help="LoRA parameter dtype. auto follows the base for training and preserves saved dtype on reload.",
    )
    parser.add_argument(
        "--amp", choices=("auto", "off"), default="auto",
        help="auto enables CUDA autocast for BF16/FP16 models in training and evaluation.",
    )
    parser.add_argument("--full-determinism", action="store_true")
    parser.add_argument(
        "--attn-implementation", choices=("auto", "eager", "sdpa"), default="auto",
        help="auto retains Transformers attention selection; eager supports strict comparisons.",
    )


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="JSON/JSONL file, URL, or glob accepted by datasets.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Dataset column containing already-formatted training text.",
    )
    parser.add_argument(
        "--data-format", choices=("text", "prompt-completion"), default="text",
        help="prompt-completion requires already chat-formatted prompt/completion strings.",
    )
    parser.add_argument(
        "--completion-only-loss", action=argparse.BooleanOptionalAction, default=False,
        help="Supervise only completion tokens; requires --data-format prompt-completion.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, and export a NanoFT LoRA adapter. "
            "The defaults use Qwen3-0.6B, but all model-specific values "
            "are command-line options."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Prepare LoRA with NanoFT and train it with TRL SFTTrainer.",
    )
    add_model_argument(train_parser)
    add_data_arguments(train_parser)
    train_parser.add_argument(
        "--eval-data-path",
        type=Path,
        default=None,
        help="Optional JSON/JSONL validation dataset for periodic loss evaluation.",
    )
    train_parser.add_argument("--eval-steps", type=int, default=1000)
    train_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    train_parser.add_argument(
        "--target-modules",
        default=DEFAULT_TARGET_MODULES,
        help="Comma-separated Linear module names; override for other architectures.",
    )
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--lora-dropout", type=float, default=0.0)
    train_parser.add_argument("--max-seq-length", type=int, default=2048)
    train_parser.add_argument("--max-samples", type=int, default=2048)
    train_parser.add_argument("--max-steps", type=int, default=100)
    train_parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs when --max-steps is -1.",
    )
    train_parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="TRL/Transformers trainer checkpoint directory to resume from.",
    )
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=10)
    train_parser.add_argument("--logging-steps", type=int, default=1)
    train_parser.add_argument("--save-steps", type=int, default=50)
    train_parser.add_argument("--seed", type=int, default=3407)
    train_parser.add_argument("--dataset-num-proc", type=int, default=1)
    train_parser.add_argument("--report-to", default="none")
    train_parser.add_argument("--save-merged", action="store_true")
    train_parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Compare base, NanoFT, PEFT, and merged model outputs.",
    )
    add_model_argument(evaluate_parser)
    add_data_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    evaluate_parser.add_argument("--native-adapter", type=Path)
    evaluate_parser.add_argument("--peft-adapter", type=Path)
    evaluate_parser.add_argument("--merged-model", type=Path)
    evaluate_parser.add_argument("--variants", default="base,native")
    evaluate_parser.add_argument("--eval-offset", type=int, default=2048)
    evaluate_parser.add_argument("--eval-samples", type=int, default=64)
    evaluate_parser.add_argument("--eval-max-length", type=int, default=1024)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=256)
    evaluate_parser.add_argument("--prompts-file", type=Path)
    evaluate_parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format string containing a {prompt} field.",
    )
    evaluate_parser.add_argument("--metrics-output", type=Path)
    evaluate_parser.add_argument("--seed", type=int, default=3407)
    evaluate_parser.add_argument("--train-samples", type=int, default=2048)
    evaluate_parser.add_argument("--train-batch-size", type=int, default=1)
    evaluate_parser.add_argument(
        "--train-gradient-accumulation", type=int, default=4
    )
    evaluate_parser.add_argument("--train-world-size", type=int, default=1)

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge a NanoFT/PEFT adapter into a standard Transformers model.",
    )
    add_model_argument(merge_parser)
    merge_parser.add_argument("--seed", type=int, default=3407)
    merge_parser.add_argument(
        "--adapter",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "adapter_native",
    )
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "merged_model",
    )
    merge_parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload the export and compare it with the in-memory merged model.",
    )
    merge_parser.add_argument(
        "--verify-prompt", default=DEFAULT_PROMPT_TEMPLATE.format(
            prompt="Briefly explain what LoRA does."
        )
    )
    return parser


def precision_for_device(
    requested_dtype: str = "auto",
    amp: str = "auto",
) -> tuple[str, torch.dtype, bool, bool]:
    device_info = detect_device()
    if requested_dtype == "auto":
        dtype = get_recommended_dtype(
            device_info.device_type,
            supports_bf16=device_info.supports_bf16,
        )
        use_bf16 = (
            device_info.device_type == "cuda"
            and dtype is torch.bfloat16
        )
        use_fp16 = (
            device_info.device_type == "cuda"
            and dtype is torch.float16
        )
        return device_info.device_type, dtype, use_bf16 and amp == "auto", use_fp16 and amp == "auto"

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested_dtype]
    if dtype is torch.bfloat16 and (
        device_info.device_type != "cuda"
        or not device_info.supports_bf16
    ):
        raise ValueError("bfloat16 requires a BF16-capable CUDA device")
    if device_info.device_type == "cpu" and dtype is not torch.float32:
        raise ValueError("The demo supports float32 only on CPU")

    # TRL mixed-precision flags are enabled on CUDA. On MPS, explicit FP16
    # means loading/training the model in FP16 without CUDA AMP.
    use_bf16 = device_info.device_type == "cuda" and dtype is torch.bfloat16
    use_fp16 = device_info.device_type == "cuda" and dtype is torch.float16
    return device_info.device_type, dtype, use_bf16 and amp == "auto", use_fp16 and amp == "auto"


def dtype_load_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Avoid dtype keyword deprecation warnings across Transformers 4 and 5."""
    major_version = int(transformers_version.split(".", 1)[0])
    return {"dtype" if major_version >= 5 else "torch_dtype": dtype}


def configure_execution(args: argparse.Namespace) -> None:
    # Set CUDA environment before seeding or inspecting/initializing a device.
    if args.full_determinism:
        os.environ.update(DETERMINISM_ENV)
        enable_full_determinism(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        set_seed(args.seed)


def model_load_kwargs(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    kwargs = dtype_load_kwargs(dtype)
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    return kwargs


def resolve_adapter_dtype(requested: str, path: Path | None = None) -> torch.dtype | None:
    if requested != "auto":
        return getattr(torch, requested)
    if path is None:
        return None
    safe_path = path / "adapter_model.safetensors"
    if safe_path.exists():
        from safetensors import safe_open

        dtype_names = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
        with safe_open(safe_path, framework="pt", device="cpu") as weights:
            dtypes = {
                dtype_names.get(weights.get_slice(key).get_dtype())
                for key in weights.keys() if "lora_A" in key or "lora_B" in key
            }
    else:
        weights = torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)
        dtypes = {value.dtype for key, value in weights.items() if "lora_A" in key or "lora_B" in key}
    if len(dtypes) != 1 or None in dtypes or not dtypes <= {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(f"Cannot infer one supported adapter dtype from {path}; specify --adapter-dtype")
    return next(iter(dtypes))


def execution_config(
    args: argparse.Namespace, model: torch.nn.Module, device: str,
    dtype: torch.dtype, trainer: Any = None,
) -> dict[str, Any]:
    parameter_dtypes: dict[str, dict[str, int]] = {"base": {}, "adapter": {}}
    for name, parameter in model.named_parameters():
        group = "adapter" if "lora_A" in name or "lora_B" in name else "base"
        counts = parameter_dtypes[group]
        counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + parameter.numel()
    attention = {
        name or "model": getattr(module.config, "_attn_implementation", None)
        for name, module in model.named_modules() if hasattr(module, "config")
    }
    _, _, bf16, fp16 = precision_for_device(args.dtype, args.amp)
    return {
        "requested": vars(args).copy(), "device": device, "model_dtype": str(dtype),
        "parameter_dtypes": parameter_dtypes, "attention_implementation": attention,
        "amp": "bf16" if bf16 else "fp16" if fp16 else "no",
        "trainer_mixed_precision": trainer.accelerator.mixed_precision if trainer is not None else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "environment": {
            key: os.environ.get(key)
            for key in (*DETERMINISM_ENV, "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE")
        },
    }


def data_processing_config(args: argparse.Namespace, tokenizer: Any, max_length: int) -> dict[str, Any]:
    return {
        "format": args.data_format, "completion_only_loss": args.completion_only_loss,
        "add_special_tokens": args.data_format == "text", "append_eos_if_missing": True,
        "eos_token": tokenizer.eos_token, "max_length": max_length,
        "truncation_side": "right", "padding_side": "right",
        "supervised_tokens": "labels[1:] != -100 (causal shift)",
    }


def tokenize_example(
    example: dict[str, Any], tokenizer: Any, args: argparse.Namespace, max_length: int,
) -> dict[str, list[int]]:
    if max_length < 2:
        raise ValueError("max length must be at least 2 for causal supervision")
    if args.completion_only_loss and args.data_format != "prompt-completion":
        raise ValueError("completion-only loss requires prompt-completion data, not text")
    fields = [args.text_column] if args.data_format == "text" else ["prompt", "completion"]
    for field in fields:
        if not isinstance(example.get(field), str) or not example[field].strip():
            raise ValueError(f"Expected a non-empty string in {field!r}")
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must define eos_token")
    prompt = "" if args.data_format == "text" else example["prompt"]
    text = example[args.text_column] if args.data_format == "text" else prompt + example["completion"]
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    # Preserve TRL's historical text handling. Chat-formatted pairs already
    # contain their special tokens and must not acquire another BOS/EOS.
    input_ids = tokenizer(text, add_special_tokens=args.data_format == "text")["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Joint tokenization does not preserve the prompt token prefix; fix the prompt/completion boundary")
    input_ids = input_ids[:max_length]
    labels = input_ids.copy()
    if args.completion_only_loss:
        labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No supervised tokens remain after causal shift and truncation")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def tokenize_dataset(dataset: Any, tokenizer: Any, args: argparse.Namespace, max_length: int) -> Any:
    return dataset.map(
        lambda row: tokenize_example(row, tokenizer, args, max_length),
        remove_columns=dataset.column_names,
        num_proc=getattr(args, "dataset_num_proc", 1),
        desc="Tokenizing and building causal labels",
    )


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_text_dataset(args: argparse.Namespace, data_path: str | Path | None = None) -> Any:
    dataset = load_dataset(
        "json", data_files={"train": str(data_path or args.data_path)}, split="train"
    )
    required = [args.text_column] if args.data_format == "text" else ["prompt", "completion"]
    if not set(required) <= set(dataset.column_names):
        raise ValueError(
            f"Expected columns {required!r}, got {dataset.column_names}"
        )
    return dataset


def clear_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


def run_train(args: argparse.Namespace) -> None:
    configure_execution(args)
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training requires TRL and Accelerate. Install compatible releases "
            "for your Transformers environment, for example: "
            "`pip install trl accelerate`."
        ) from exc

    started_at = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_text_dataset(args)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    eval_dataset = None
    if args.eval_data_path is not None:
        eval_dataset = load_text_dataset(args, args.eval_data_path)

    device, model_dtype, use_bf16, use_fp16 = precision_for_device(args.dtype, args.amp)
    print(f"Device: {device}; model dtype: {model_dtype}")
    tokenizer = load_tokenizer(args.model_name)
    dataset = tokenize_dataset(dataset, tokenizer, args, args.max_seq_length)
    if eval_dataset is not None:
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args, args.max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_load_kwargs(args, model_dtype)
    )
    model.to(device)

    target_modules = [
        name.strip() for name in args.target_modules.split(",") if name.strip()
    ]
    if not target_modules:
        raise ValueError("target-modules must contain at least one module name")
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        base_model_name_or_path=args.model_name,
    )
    model = prepare_model_for_training(
        model, lora_config, adapter_dtype=resolve_adapter_dtype(args.adapter_dtype)
    )
    model.config.use_cache = not args.gradient_checkpointing
    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        args=SFTConfig(
            output_dir=str(args.output_root / "trainer"),
            dataset_text_field=args.text_column,
            dataset_kwargs={"skip_prepare_dataset": True},
            completion_only_loss=args.completion_only_loss,
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=args.eval_steps,
            prediction_loss_only=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            optim="adamw_torch",
            bf16=use_bf16,
            fp16=use_fp16,
            full_determinism=args.full_determinism,
            tf32=False if args.full_determinism and device == "cuda" else None,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_num_proc=args.dataset_num_proc,
            seed=args.seed,
            report_to=args.report_to,
        ),
    )
    execution = execution_config(args, trainer.model, device, model_dtype, trainer)
    execution["data_processing"] = data_processing_config(args, tokenizer, args.max_seq_length)
    (args.output_root / "execution_config.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    trained_model = trainer.model
    if hasattr(trained_model, "gradient_checkpointing_disable"):
        trained_model.gradient_checkpointing_disable()
    trained_model.config.use_cache = True

    native_adapter_dir = args.output_root / "adapter_native"
    peft_adapter_dir = args.output_root / "adapter_peft"
    merged_dir = args.output_root / "merged_model"
    save_adapter(trained_model, str(native_adapter_dir), lora_config)
    save_peft_adapter(trained_model, str(peft_adapter_dir), lora_config)
    if args.save_merged:
        save_merged_model(trained_model, str(merged_dir), tokenizer=tokenizer)

    metrics = {
        "model": args.model_name,
        "data": args.data_path,
        "text_column": args.text_column,
        "samples": len(dataset),
        "eval_data_path": (
            str(args.eval_data_path) if args.eval_data_path is not None else None
        ),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "eval_steps": args.eval_steps,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
        "global_step": trainer.state.global_step,
        "target_modules": target_modules,
        "device": device,
        "model_dtype": str(model_dtype),
        "requested_dtype": args.dtype,
        "native_adapter_dir": str(native_adapter_dir),
        "peft_adapter_dir": str(peft_adapter_dir),
        "merged_dir": str(merged_dir) if args.save_merged else None,
        "runtime_sec": time.time() - started_at,
        "train_metrics": train_result.metrics,
        "execution": execution,
        "loss_reduction": "TRL/Transformers Trainer default aggregation; see train_metrics",
    }
    metrics_path = args.output_root / "run_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Metrics written to {metrics_path}")


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(variants) - SUPPORTED_VARIANTS)
    if invalid:
        raise ValueError(
            f"Unsupported variants {invalid}; choose from "
            f"{sorted(SUPPORTED_VARIANTS)}"
        )
    if not variants:
        raise ValueError("At least one evaluation variant is required")
    return list(dict.fromkeys(variants))


def print_step_recommendation(args: argparse.Namespace) -> None:
    effective_batch = (
        args.train_batch_size
        * args.train_gradient_accumulation
        * args.train_world_size
    )
    if effective_batch <= 0 or args.train_samples <= 0:
        raise ValueError("Training sample and batch values must be positive")
    steps_per_epoch = math.ceil(args.train_samples / effective_batch)
    print("\nTraining-step guidance")
    print(f"  effective batch:     {effective_batch}")
    print(f"  steps per epoch:     {steps_per_epoch}")
    print(f"  smoke test:          {min(20, steps_per_epoch)} steps")
    print(
        f"  first useful range:  "
        f"{max(1, math.ceil(steps_per_epoch * 0.5))}-"
        f"{steps_per_epoch * 2} steps (~0.5-2 epochs)"
    )
    print("  select the final checkpoint using held-out loss and generations.\n")


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt") or item.get("text")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{path}:{line_number} needs a non-empty prompt/text field"
                )
            prompts.append(prompt.strip())
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_held_out_examples(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, list[int]]]:
    if args.eval_samples <= 0:
        return []
    if args.eval_offset < 0:
        raise ValueError("eval-offset must be non-negative")
    dataset = load_text_dataset(args)
    if args.eval_offset >= len(dataset):
        raise ValueError(
            f"eval-offset {args.eval_offset} is outside {len(dataset)} rows"
        )
    end = min(args.eval_offset + args.eval_samples, len(dataset))
    return [
        tokenize_example(row, tokenizer, args, args.eval_max_length)
        for row in dataset.select(range(args.eval_offset, end))
    ]


def resolve_variant_path(
    args: argparse.Namespace, variant: str
) -> Path | None:
    if variant == "base":
        return None
    explicit = {
        "native": args.native_adapter,
        "peft": args.peft_adapter,
        "merged": args.merged_model,
    }
    defaults = {
        "native": args.output_root / "adapter_native",
        "peft": args.output_root / "adapter_peft",
        "merged": args.output_root / "merged_model",
    }
    path = explicit[variant] or defaults[variant]
    if not path.exists():
        raise FileNotFoundError(f"{variant} weights not found at {path}")
    return path


def load_evaluation_model(
    args: argparse.Namespace,
    variant: str,
    device: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    variant_path = resolve_variant_path(args, variant)
    source = str(variant_path) if variant == "merged" else args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        source, **model_load_kwargs(args, dtype)
    )
    model.to(device)
    if variant == "native":
        model = load_adapter(
            model, str(variant_path),
            adapter_dtype=resolve_adapter_dtype(args.adapter_dtype, variant_path),
        )
    elif variant == "peft":
        try:
            from peft import PeftConfig, get_peft_model
        except ImportError as exc:
            raise SystemExit("PEFT evaluation requires `pip install peft`.") from exc
        adapter_dtype = resolve_adapter_dtype(args.adapter_dtype, variant_path)
        config = PeftConfig.from_pretrained(variant_path)
        config.inference_mode = True
        model = get_peft_model(model, config, autocast_adapter_dtype=False)
        # Cast the empty adapter containers before loading, avoiding an
        # irreversible FP32 -> BF16 -> FP32 round trip in PEFT comparison.
        for name, parameter in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.data = parameter.data.to(adapter_dtype)
        model.load_adapter(
            str(variant_path), adapter_name="default", is_trainable=False,
            autocast_adapter_dtype=False,
        )
    model.eval()
    model.config.use_cache = True
    return model


def inference_autocast(device: str, dtype: torch.dtype, amp: str) -> Any:
    return torch.autocast(
        device_type=device, dtype=dtype,
        enabled=amp == "auto" and device == "cuda" and dtype in (torch.bfloat16, torch.float16),
    )


def evaluate_loss(
    model: torch.nn.Module, tokenizer: Any, examples: list[dict[str, list[int]]],
    device: str, dtype: torch.dtype, amp: str,
) -> dict[str, float] | None:
    if not examples:
        return None
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
    with torch.no_grad():
        for example in examples:
            encoded = collator([example]).to(device)
            token_count = int((encoded["labels"][:, 1:] != -100).sum().item())
            with inference_autocast(device, dtype, amp):
                loss = model(**encoded).loss
            total_negative_log_likelihood += float(loss.item()) * token_count
            total_tokens += token_count
    mean_loss = total_negative_log_likelihood / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "tokens": total_tokens,
        "samples": len(examples),
    }


def generate_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    prompt_template: str,
    max_new_tokens: int,
    add_special_tokens: bool = True,
) -> list[dict[str, str]]:
    if "{prompt}" not in prompt_template:
        raise ValueError("prompt-template must contain {prompt}")
    device = next(model.parameters()).device
    responses = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt_template.format(prompt=prompt), return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        responses.append(
            {
                "prompt": prompt,
                "response": tokenizer.decode(
                    generated, skip_special_tokens=True
                ).strip(),
            }
        )
    return responses


def run_evaluate(args: argparse.Namespace) -> None:
    configure_execution(args)
    variants = parse_variants(args.variants)
    print_step_recommendation(args)
    prompts = load_prompts(args.prompts_file)
    device, dtype, _, _ = precision_for_device(args.dtype, args.amp)
    tokenizer = load_tokenizer(args.model_name)
    held_out_examples = load_held_out_examples(args, tokenizer)
    results: dict[str, Any] = {
        "model": args.model_name,
        "variants": variants,
        "eval_offset": args.eval_offset,
        "eval_samples": len(held_out_examples),
        "data_processing": data_processing_config(args, tokenizer, args.eval_max_length),
        "loss_reduction": "sum of per-example mean loss times supervised tokens / total supervised tokens",
        "generation_add_special_tokens": args.data_format == "text",
        "results": {},
    }
    for variant in variants:
        print(f"\nLoading {variant} on {device} with {dtype}...")
        model = load_evaluation_model(args, variant, device, dtype)
        loss_metrics = evaluate_loss(
            model, tokenizer, held_out_examples, device, dtype, args.amp
        )
        with inference_autocast(device, dtype, args.amp):
            generations = generate_responses(
                model, tokenizer, prompts, args.prompt_template, args.max_new_tokens,
                add_special_tokens=args.data_format == "text",
            )
        results["results"][variant] = {
            "language_model": loss_metrics,
            "generations": generations,
            "execution": execution_config(args, model, device, dtype),
        }
        if loss_metrics:
            print(
                f"{variant}: loss={loss_metrics['loss']:.4f}, "
                f"perplexity={loss_metrics['perplexity']:.4f}"
            )
        for item in generations:
            print(f"\n[{variant}] {item['prompt']}\n{item['response']}")
        del model
        clear_device_cache()

    metrics_path = (
        args.metrics_output
        or args.output_root / "evaluation_results.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nEvaluation results written to {metrics_path}")


def verification_logits(
    model: torch.nn.Module, tokenizer: Any, prompt: str
) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt").to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    return logits.detach().float().cpu()


def difference_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    difference = (candidate - reference).abs()
    return {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "top1_agreement": float(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1))
            .float()
            .mean()
        ),
    }


def run_merge(args: argparse.Namespace) -> None:
    configure_execution(args)
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter}")
    if args.output_dir.resolve() == args.adapter.resolve():
        raise ValueError("output-dir must differ from the adapter directory")

    device, dtype, _, _ = precision_for_device(args.dtype, args.amp)
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **model_load_kwargs(args, dtype)
    )
    model.to(device)
    model = load_adapter(
        model, str(args.adapter),
        adapter_dtype=resolve_adapter_dtype(args.adapter_dtype, args.adapter),
    )
    model.eval()
    model.config.use_cache = True
    execution = execution_config(args, model, device, dtype)
    execution["verification_add_special_tokens"] = True
    with inference_autocast(device, dtype, args.amp):
        adapter_logits = (
            verification_logits(model, tokenizer, args.verify_prompt)
            if args.verify else None
        )

    save_merged_model(
        model,
        str(args.output_dir),
        tokenizer=tokenizer,
        inplace=True,
    )
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("LoRA layers remain after merge/export")
    if any(
        "lora_A" in key or "lora_B" in key for key in model.state_dict()
    ):
        raise RuntimeError("LoRA parameters remain after merge/export")
    print(f"Standard Transformers model written to {args.output_dir}")
    execution["merged_parameter_dtypes"] = execution_config(args, model, device, dtype)["parameter_dtypes"]
    (args.output_dir / "execution_config.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if adapter_logits is None:
        return

    with inference_autocast(device, dtype, args.amp):
        merged_logits = verification_logits(model, tokenizer, args.verify_prompt)
    merge_metrics = difference_metrics(adapter_logits, merged_logits)
    print(
        "Adapter -> in-memory merged drift: "
        f"max_abs_diff={merge_metrics['max_abs_diff']:.6g}, "
        f"mean_abs_diff={merge_metrics['mean_abs_diff']:.6g}, "
        f"top1_agreement={merge_metrics['top1_agreement']:.2%}"
    )
    del model
    clear_device_cache()

    restored = AutoModelForCausalLM.from_pretrained(
        args.output_dir, **model_load_kwargs(args, dtype)
    )
    restored.to(device)
    restored.eval()
    with inference_autocast(device, dtype, args.amp):
        reloaded_logits = verification_logits(restored, tokenizer, args.verify_prompt)
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    reload_metrics = difference_metrics(merged_logits, reloaded_logits)
    if not torch.allclose(
        reloaded_logits,
        merged_logits,
        atol=tolerance,
        rtol=tolerance,
    ):
        raise RuntimeError(
            "Reloaded model does not match the in-memory merged model: "
            f"{reload_metrics}, tolerance={tolerance}"
        )
    (args.output_dir / "merge_verification.json").write_text(
        json.dumps({
            "adapter_to_merged": merge_metrics,
            "merged_to_reloaded": reload_metrics,
            "reload_tolerance": tolerance,
        }, indent=2), encoding="utf-8",
    )
    print(
        "Verification passed: reloaded model matches in-memory merged model; "
        f"max_abs_diff={reload_metrics['max_abs_diff']:.6g}, "
        f"mean_abs_diff={reload_metrics['mean_abs_diff']:.6g}, "
        f"top1_agreement={reload_metrics['top1_agreement']:.2%}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
