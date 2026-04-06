"""Evaluate LoRA fine-tuning: compare Base model vs LoRA model side by side.

Metrics: SequenceMatcher similarity + ROUGE-L + BLEU-4.

Usage:
    python evaluate.py
    python evaluate.py --num-samples 20 --data-path /path/to/alpaca_gpt4_data_zh.json
    python evaluate.py --adapter-path ./output/adapter --merged-path ./output/merged
"""

import argparse
import gc
import json
import os
import random
from collections import Counter
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

from nanoft import LoRAConfig, apply_lora, load_adapter, merge_lora_weights


# ── Metrics ───────────────────────────────────────────────────


def text_similarity(text1: str, text2: str) -> float:
    """SequenceMatcher ratio (0-1)."""
    return SequenceMatcher(None, text1, text2).ratio()


def _tokenize_zh(text: str) -> list[str]:
    return [ch for ch in text if ch.strip()]


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu4(hypothesis: str, reference: str) -> float:
    hyp = _tokenize_zh(hypothesis)
    ref = _tokenize_zh(reference)
    if not hyp or not ref:
        return 0.0
    bp = min(1.0, len(hyp) / len(ref))
    log_avg = 0.0
    for n in range(1, 5):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        if not hyp_ng:
            return 0.0
        clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        if clipped == 0 and n > 1:
            clipped = 1
        import math
        log_avg += math.log(clipped / total + 1e-10)
    return bp * math.exp(log_avg / 4)


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    hyp = _tokenize_zh(hypothesis)
    ref = _tokenize_zh(reference)
    if not hyp or not ref:
        return 0.0
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Inference ─────────────────────────────────────────────────


def format_prompt(instruction: str, input_text: str = "") -> str:
    parts = ["Human: " + instruction]
    if input_text.strip():
        parts.append(input_text.strip())
    return "\n".join(parts) + "\n\nAssistant: "


def generate(model, tokenizer, instruction: str, input_text: str = "",
             max_length: int = 256) -> str:
    prompt = format_prompt(instruction, input_text)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    return response


# ── Model loading ─────────────────────────────────────────────


def load_base_model(model_dir: str):
    print("[Loading base model...]")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def load_lora_model(model_dir: str, adapter_path: str):
    print("[Loading LoRA model...]")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )

    config = LoRAConfig.load(os.path.join(adapter_path, "adapter_config.json"))
    model = apply_lora(model, config)
    model = load_adapter(model, adapter_path)
    merge_lora_weights(model)
    model.eval()
    return model, tokenizer


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Evaluation ────────────────────────────────────────────────


def evaluate_model(model, tokenizer, test_data):
    results = []
    sim_total = rouge_total = bleu_total = 0.0

    for item in test_data:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        reference = item["output"]

        prediction = generate(model, tokenizer, instruction, input_text)

        sim = text_similarity(prediction, reference)
        rl = rouge_l_f1(prediction, reference)
        b4 = bleu4(prediction, reference)
        sim_total += sim
        rouge_total += rl
        bleu_total += b4

        results.append({
            "instruction": instruction,
            "input": input_text,
            "reference": reference[:200] + "..." if len(reference) > 200 else reference,
            "prediction": prediction[:200] + "..." if len(prediction) > 200 else prediction,
            "similarity": sim,
            "rouge_l_f1": rl,
            "bleu_4": b4,
        })

    n = len(test_data)
    return results, sim_total / n, rouge_total / n, bleu_total / n


def print_comparison_table(base_results, lora_results, base_scores, lora_scores):
    base_sim, base_rouge, base_bleu = base_scores
    lora_sim, lora_rouge, lora_bleu = lora_scores

    print("\n" + "=" * 90)
    print("Evaluation Results: Base vs LoRA")
    print("=" * 90)
    print(f"\n{'Metric':<25} {'Base':>10} {'LoRA':>10} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Similarity (SeqMatcher)':<25} {base_sim:>9.2%} {lora_sim:>9.2%} {lora_sim - base_sim:>+9.2%}")
    print(f"{'ROUGE-L F1':<25} {base_rouge:>9.3f} {lora_rouge:>9.3f} {lora_rouge - base_rouge:>+.3f}")
    print(f"{'BLEU-4':<25} {base_bleu:>9.3f} {lora_bleu:>9.3f} {lora_bleu - base_bleu:>+.3f}")

    print("\n" + "-" * 90)
    print("Detail (top 5):")
    print("-" * 90)
    for i, (b, l) in enumerate(zip(base_results[:5], lora_results[:5])):
        print(f"\n[{i + 1}] {b['instruction']}")
        if b["input"]:
            print(f"    Input: {b['input']}")
        print(f"    Reference: {b['reference']}")
        print(f"    Base  (sim={b['similarity']:.2%}): {b['prediction']}")
        print(f"    LoRA  (sim={l['similarity']:.2%}): {l['prediction']}")
        print("-" * 90)


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs LoRA model")
    parser.add_argument("--model-path", default=None, help="Base model path")
    parser.add_argument("--adapter-path", default="./output/adapter")
    parser.add_argument("--data-path", default="alpaca_gpt4_data_zh.json")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation_results.json", help="Save results to JSON")
    args = parser.parse_args()

    # Load test data
    with open(args.data_path, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    test_data = rng.sample(data, min(args.num_samples, len(data)))
    print(f"Loaded {len(test_data)} test samples")

    # Model path
    if args.model_path is None:
        args.model_path = snapshot_download("Qwen/Qwen2.5-3B-Instruct")

    # Evaluate Base
    print("\n" + "=" * 60)
    print("Evaluating Base model...")
    print("=" * 60)
    model, tokenizer = load_base_model(args.model_path)
    base_results, base_sim, base_rouge, base_bleu = evaluate_model(model, tokenizer, test_data)
    clear_gpu()

    # Evaluate LoRA
    print("\n" + "=" * 60)
    print("Evaluating LoRA model...")
    print("=" * 60)
    model, tokenizer = load_lora_model(args.model_path, args.adapter_path)
    lora_results, lora_sim, lora_rouge, lora_bleu = evaluate_model(model, tokenizer, test_data)
    clear_gpu()

    # Print comparison
    print_comparison_table(
        base_results, lora_results,
        (base_sim, base_rouge, base_bleu),
        (lora_sim, lora_rouge, lora_bleu),
    )

    # Save results
    output = {
        "base_score": {"similarity": base_sim, "rouge_l_f1": base_rouge, "bleu_4": base_bleu},
        "lora_score": {"similarity": lora_sim, "rouge_l_f1": lora_rouge, "bleu_4": lora_bleu},
        "improvement": {
            "similarity": lora_sim - base_sim,
            "rouge_l_f1": lora_rouge - base_rouge,
            "bleu_4": lora_bleu - base_bleu,
        },
        "comparisons": [
            {
                "instruction": b["instruction"],
                "reference": b["reference"],
                "base_prediction": b["prediction"],
                "lora_prediction": l["prediction"],
                "base_similarity": b["similarity"],
                "lora_similarity": l["similarity"],
                "base_rouge_l_f1": b["rouge_l_f1"],
                "lora_rouge_l_f1": l["rouge_l_f1"],
                "base_bleu_4": b["bleu_4"],
                "lora_bleu_4": l["bleu_4"],
            }
            for b, l in zip(base_results, lora_results)
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
