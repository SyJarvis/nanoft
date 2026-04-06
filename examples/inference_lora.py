"""Example: load a trained LoRA adapter for inference and evaluate with ROUGE-L / BLEU.

Usage:
    python inference_lora.py --method adapter
    python inference_lora.py --method merged
    python inference_lora.py --method both --num-samples 10
    python inference_lora.py --method adapter --data-path /path/to/alpaca_gpt4_data_zh.json
"""

import argparse
import json
import random
import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download

from nanoft import LoRAConfig, apply_lora, load_adapter, merge_lora_weights


# ── Metrics ───────────────────────────────────────────────────


def _tokenize_zh(text: str) -> list[str]:
    """Simple char-level tokenization for Chinese text."""
    # Strip whitespace, keep non-space chars as tokens
    return [ch for ch in text if ch.strip()]


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> dict[str, float]:
    """Compute BLEU-1..4 between hypothesis and reference."""
    hyp_tokens = _tokenize_zh(hypothesis)
    ref_tokens = _tokenize_zh(reference)

    if not hyp_tokens or not ref_tokens:
        return {f"bleu-{i}": 0.0 for i in range(1, max_n + 1)}

    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0

    scores = {}
    log_avg = 0.0
    smooth = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        if not hyp_ngrams:
            scores[f"bleu-{n}"] = 0.0
            continue
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        # Smooth for higher n-grams
        if clipped == 0 and n > 1:
            smooth += 1
            clipped = 1
            total = max(total, 1)
        precision = clipped / total if total > 0 else 0.0
        scores[f"bleu-{n}"] = bp * precision

    return scores


def rouge_l_score(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L (F1, Precision, Recall) based on LCS."""
    hyp_tokens = _tokenize_zh(hypothesis)
    ref_tokens = _tokenize_zh(reference)

    if not hyp_tokens or not ref_tokens:
        return {"rouge-l/precision": 0.0, "rouge-l/recall": 0.0, "rouge-l/f1": 0.0}

    # LCS length via DP
    m, n = len(hyp_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "rouge-l/precision": precision,
        "rouge-l/recall": recall,
        "rouge-l/f1": f1,
    }


def evaluate(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute all metrics for a single prediction."""
    metrics = {}
    metrics.update(rouge_l_score(hypothesis, reference))
    metrics.update(bleu_score(hypothesis, reference))
    return metrics


def print_score_table(results: list[dict]) -> None:
    """Print per-sample and average scores."""
    # Header
    print(f"  {'#':>3}  {'R-L/P':>6}  {'R-L/R':>6}  {'R-L/F1':>6}  {'B-1':>6}  {'B-2':>6}  {'B-3':>6}  {'B-4':>6}")
    print("  " + "-" * 60)

    for i, r in enumerate(results):
        print(
            f"  {i + 1:>3}  "
            f"{r['rouge-l/precision']:>6.3f}  "
            f"{r['rouge-l/recall']:>6.3f}  "
            f"{r['rouge-l/f1']:>6.3f}  "
            f"{r['bleu-1']:>6.3f}  "
            f"{r['bleu-2']:>6.3f}  "
            f"{r['bleu-3']:>6.3f}  "
            f"{r['bleu-4']:>6.3f}"
        )

    # Average
    keys = ["rouge-l/precision", "rouge-l/recall", "rouge-l/f1",
            "bleu-1", "bleu-2", "bleu-3", "bleu-4"]
    print("  " + "-" * 60)
    avgs = []
    for k in keys:
        avg = sum(r[k] for r in results) / len(results)
        avgs.append(f"{avg:>6.3f}")
    print(f"  AVG  " + "  ".join(avgs))
    print()


# ── Inference helpers ─────────────────────────────────────────


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format prompt to match the training data format in train_lora.py."""
    parts = ["Human: " + instruction]
    if input_text.strip():
        parts.append(input_text.strip())
    return "\n".join(parts) + "\n\nAssistant: "


def load_test_samples(data_path: str, num_samples: int = 5, seed: int = 42):
    """Load test samples from alpaca-format dataset."""
    with open(data_path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    samples = rng.sample(data, min(num_samples, len(data)))
    return samples


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Run a single generation and return decoded text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_inference(model, tokenizer, samples, label: str) -> None:
    """Run inference on test samples, compute metrics, print results."""
    results = []

    for i, sample in enumerate(samples):
        prompt = format_prompt(sample["instruction"], sample.get("input", ""))
        generated = generate(model, tokenizer, prompt)
        expected = sample["output"]

        metrics = evaluate(generated, expected)
        results.append(metrics)

        print(f"[{label} #{i + 1}]")
        print(f"  Q: {sample['instruction']}")
        if sample.get("input", "").strip():
            print(f"  Input: {sample['input']}")
        print(f"  Expected:  {expected[:150]}...")
        print(f"  Generated: {generated[:150]}...")
        print(f"  ROUGE-L F1: {metrics['rouge-l/f1']:.3f} | BLEU-4: {metrics['bleu-4']:.3f}")
        print()

    print(f"[{label} Score Summary]")
    print_score_table(results)


# ── Method 1: Load adapter + merge ────────────────────────────


def infer_with_adapter(base_path: str, adapter_path: str, samples) -> None:
    """Load base model, inject LoRA, load adapter weights, merge, then infer."""
    print("=" * 60)
    print("Method 1: Load adapter -> merge -> inference")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, device_map="auto", trust_remote_code=True,
    )

    config = LoRAConfig.load(f"{adapter_path}/adapter_config.json")
    print(f"  LoRA config: r={config.r}, alpha={config.lora_alpha}, "
          f"targets={config.get_target_modules()}")

    model = apply_lora(model, config)
    model = load_adapter(model, adapter_path)
    print("  Adapter loaded")

    merge_lora_weights(model)
    model.eval()
    print("  LoRA merged into base weights\n")

    run_inference(model, tokenizer, samples, "Adapter")


# ── Method 2: Load pre-merged model ───────────────────────────


def infer_with_merged(merged_path: str, samples) -> None:
    """Load a merged model (LoRA already folded in) and infer directly."""
    print("=" * 60)
    print("Method 2: Load merged model -> inference")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(merged_path)
    model = AutoModelForCausalLM.from_pretrained(
        merged_path, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("  Merged model loaded\n")

    run_inference(model, tokenizer, samples, "Merged")


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LoRA inference with ROUGE-L / BLEU evaluation")
    parser.add_argument(
        "--method",
        choices=["adapter", "merged", "both"],
        default="both",
        help="Inference method to run (default: both)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Base model path (default: auto-download Qwen3.5-2B)",
    )
    parser.add_argument(
        "--adapter-path", default="./output/adapter",
        help="Path to saved adapter (default: ./output/adapter)",
    )
    parser.add_argument(
        "--merged-path", default="./output/merged",
        help="Path to merged model (default: ./output/merged)",
    )
    parser.add_argument(
        "--data-path", default="../../alpaca_gpt4_data_zh.json",
        help="Path to alpaca test dataset (default: ../../alpaca_gpt4_data_zh.json)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of test samples to run (default: 5)",
    )
    args = parser.parse_args()

    samples = load_test_samples(args.data_path, args.num_samples)
    print(f"Loaded {len(samples)} test samples from {args.data_path}\n")

    if args.model_path is None:
        args.model_path = snapshot_download("Qwen/Qwen3.5-2B")

    if args.method in ("adapter", "both"):
        infer_with_adapter(args.model_path, args.adapter_path, samples)

    if args.method in ("merged", "both"):
        infer_with_merged(args.merged_path, samples)


if __name__ == "__main__":
    main()
