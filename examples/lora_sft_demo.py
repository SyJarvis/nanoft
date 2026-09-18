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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


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
        return device_info.device_type, dtype, use_bf16, use_fp16

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
    return device_info.device_type, dtype, use_bf16, use_fp16


def dtype_load_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Avoid dtype keyword deprecation warnings across Transformers 4 and 5."""
    major_version = int(transformers_version.split(".", 1)[0])
    return {"dtype" if major_version >= 5 else "torch_dtype": dtype}


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_text_dataset(args: argparse.Namespace) -> Any:
    dataset = load_dataset(
        "json", data_files={"train": args.data_path}, split="train"
    )
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Expected column {args.text_column!r}, got {dataset.column_names}"
        )
    return dataset


def clear_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


def run_train(args: argparse.Namespace) -> None:
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training requires TRL and Accelerate. Install compatible releases "
            "for your Transformers environment, for example: "
            "`pip install trl accelerate`."
        ) from exc

    set_seed(args.seed)
    started_at = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_text_dataset(args)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    device, model_dtype, use_bf16, use_fp16 = precision_for_device(args.dtype)
    print(f"Device: {device}; model dtype: {model_dtype}")
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **dtype_load_kwargs(model_dtype)
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
    model = prepare_model_for_training(model, lora_config)
    model.config.use_cache = not args.gradient_checkpointing
    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(args.output_root / "trainer"),
            dataset_text_field=args.text_column,
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            optim="adamw_torch",
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_num_proc=args.dataset_num_proc,
            seed=args.seed,
            report_to=args.report_to,
        ),
    )
    train_result = trainer.train()
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
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "target_modules": target_modules,
        "device": device,
        "model_dtype": str(model_dtype),
        "requested_dtype": args.dtype,
        "native_adapter_dir": str(native_adapter_dir),
        "peft_adapter_dir": str(peft_adapter_dir),
        "merged_dir": str(merged_dir) if args.save_merged else None,
        "runtime_sec": time.time() - started_at,
        "train_metrics": train_result.metrics,
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


def load_held_out_texts(args: argparse.Namespace) -> list[str]:
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
        text
        for text in dataset.select(range(args.eval_offset, end))[
            args.text_column
        ]
        if isinstance(text, str) and text.strip()
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
        source, **dtype_load_kwargs(dtype)
    )
    model.to(device)
    if variant == "native":
        model = load_adapter(model, str(variant_path))
    elif variant == "peft":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("PEFT evaluation requires `pip install peft`.") from exc
        model = PeftModel.from_pretrained(model, str(variant_path))
    model.eval()
    model.config.use_cache = True
    return model


def evaluate_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
) -> dict[str, float] | None:
    if not texts:
        return None
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            token_count = max(
                int(encoded["attention_mask"].sum().item()) - 1, 1
            )
            loss = model(**encoded, labels=encoded["input_ids"]).loss
            total_negative_log_likelihood += float(loss.item()) * token_count
            total_tokens += token_count
    mean_loss = total_negative_log_likelihood / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "tokens": total_tokens,
        "samples": len(texts),
    }


def generate_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    prompt_template: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    if "{prompt}" not in prompt_template:
        raise ValueError("prompt-template must contain {prompt}")
    device = next(model.parameters()).device
    responses = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt_template.format(prompt=prompt), return_tensors="pt"
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
    set_seed(args.seed)
    variants = parse_variants(args.variants)
    print_step_recommendation(args)
    prompts = load_prompts(args.prompts_file)
    held_out_texts = load_held_out_texts(args)
    device, dtype, _, _ = precision_for_device(args.dtype)
    tokenizer = load_tokenizer(args.model_name)
    results: dict[str, Any] = {
        "model": args.model_name,
        "variants": variants,
        "eval_offset": args.eval_offset,
        "eval_samples": len(held_out_texts),
        "results": {},
    }
    for variant in variants:
        print(f"\nLoading {variant} on {device} with {dtype}...")
        model = load_evaluation_model(args, variant, device, dtype)
        loss_metrics = evaluate_loss(
            model, tokenizer, held_out_texts, args.eval_max_length
        )
        generations = generate_responses(
            model,
            tokenizer,
            prompts,
            args.prompt_template,
            args.max_new_tokens,
        )
        results["results"][variant] = {
            "language_model": loss_metrics,
            "generations": generations,
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
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
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
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter}")
    if args.output_dir.resolve() == args.adapter.resolve():
        raise ValueError("output-dir must differ from the adapter directory")

    device, dtype, _, _ = precision_for_device(args.dtype)
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, **dtype_load_kwargs(dtype)
    )
    model.to(device)
    model = load_adapter(model, str(args.adapter))
    model.eval()
    model.config.use_cache = True
    adapter_logits = (
        verification_logits(model, tokenizer, args.verify_prompt)
        if args.verify
        else None
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
    if adapter_logits is None:
        return

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
        args.output_dir, **dtype_load_kwargs(dtype)
    )
    restored.to(device)
    restored.eval()
    reloaded_logits = verification_logits(
        restored, tokenizer, args.verify_prompt
    )
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
