"""Example: fine-tune with NanoFT, driven by train_config.json.

Usage:
    python train_lora.py
    python train_lora.py --config my_config.json
"""

import argparse
import json
import os

from nanoft import (
    LoRAConfig,
    save_adapter,
    save_merged_model,
    prepare_model_for_training,
    print_trainable_parameters,
    detect_device,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from modelscope import snapshot_download


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.json", help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    # ── 1. 下载或定位模型 ───────────────────────────────────
    model_dir = cfg["model_name"]
    if not os.path.exists(model_dir):
        model_dir = snapshot_download(model_dir)

    # ── 2. 加载 tokenizer 和模型 ─────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", trust_remote_code=True
    )

    # ── 3. 配置 LoRA 并注入 ─────────────────────────────────
    config = LoRAConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        base_model_name_or_path=model_dir,
    )

    model = prepare_model_for_training(model, config)
    print_trainable_parameters(model)

    # ── 4. 加载并处理数据集 ──────────────────────────────────
    dataset = load_dataset("json", data_files=data_cfg["data_path"])
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        dataset["train"] = dataset["train"].select(
            range(min(max_samples, len(dataset["train"])))
        )
    dataset = dataset["train"].train_test_split(test_size=data_cfg["test_size"])

    max_length = train_cfg["max_length"]
    data_format = data_cfg.get("format", "alpaca")
    text_column = data_cfg.get("text_column", "text")

    def process_alpaca(example):
        instruction = tokenizer(
            "\n".join(["Human: " + example["instruction"], example["input"]]).strip()
            + "\n\nAssistant: "
        )
        response = tokenizer(example["output"] + tokenizer.eos_token)
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def process_text(example):
        encoded = tokenizer(
            example[text_column] + tokenizer.eos_token,
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    if data_format == "alpaca":
        process_func = process_alpaca
    elif data_format == "text":
        process_func = process_text
    else:
        raise ValueError("data.format must be 'alpaca' or 'text'")

    tokenized_dataset = dataset.map(
        process_func, remove_columns=dataset["train"].column_names
    )

    # ── 5. 训练 ──────────────────────────────────────────────
    info = detect_device()
    is_bf16 = train_cfg.get(
        "bf16",
        info.device_type == "cuda" and info.supports_bf16,
    )
    is_fp16 = train_cfg.get(
        "fp16",
        not is_bf16 and info.device_type != "cpu",
    )

    train_args = TrainingArguments(
        output_dir=out_cfg["output_dir"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        bf16=is_bf16,
        fp16=is_fp16,
        logging_steps=train_cfg["logging_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        save_total_limit=train_cfg["save_total_limit"],
        save_steps=train_cfg["save_steps"],
        max_steps=train_cfg.get("max_steps", -1),
        eval_strategy=train_cfg.get("eval_strategy", "no"),
        eval_steps=train_cfg.get("eval_steps"),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # ── 6. 保存 adapter（PEFT 兼容格式）──────────────────────
    save_adapter(model, out_cfg["adapter_dir"], config)
    print(f"Adapter saved to {out_cfg['adapter_dir']}")

    # 保存合并后的完整模型
    save_merged_model(model, out_cfg["merged_dir"], tokenizer=tokenizer)
    print(f"Merged model saved to {out_cfg['merged_dir']}")


if __name__ == "__main__":
    main()
