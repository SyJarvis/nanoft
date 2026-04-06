"""Example: fine-tune Qwen3.5-2B with NanoFT."""

from nanoft import (
    LoRAConfig,
    apply_lora,
    merge_lora_weights,
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

# ── 1. 下载模型 ──────────────────────────────────────────────
model_dir = snapshot_download("Qwen/Qwen3.5-2B")

# ── 2. 加载 tokenizer 和模型 ─────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, device_map="auto", trust_remote_code=True
)

# ── 3. 配置 LoRA 并注入 ─────────────────────────────────────
config = LoRAConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"],
    base_model_name_or_path=model_dir,
)

model = prepare_model_for_training(model, config)
print_trainable_parameters(model)

# ── 4. 加载并处理数据集 ──────────────────────────────────────
dataset = load_dataset("json", data_files="alpaca_gpt4_data_zh.json")
dataset = dataset["train"].train_test_split(test_size=0.1)


def process_func(example):
    MAX_LENGTH = 256
    instruction = tokenizer(
        "\n".join(["Human: " + example["instruction"], example["input"]]).strip()
        + "\n\nAssistant: "
    )
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_dataset = dataset.map(
    process_func, remove_columns=dataset["train"].column_names
)

# ── 5. 训练 ──────────────────────────────────────────────────
info = detect_device()
is_bf16 = info.device_type == "cuda" and info.supports_bf16

args = TrainingArguments(
    output_dir="./output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    bf16=is_bf16,
    fp16=not is_bf16 and info.device_type != "cpu",
    logging_steps=100,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# ── 6. 保存 adapter（PEFT 兼容格式）──────────────────────────
save_adapter(model, "./output/adapter", config)
print("Adapter saved to ./output/adapter/")

# 保存合并后的完整模型（可选）
merge_lora_weights(model)
save_merged_model(model, "./output/merged", tokenizer=tokenizer)
print("Merged model saved to ./output/merged/")
