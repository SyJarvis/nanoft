import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("trl")
pytest.importorskip("peft")
from datasets import Dataset
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

from nanoft import LoRAConfig, load_adapter, prepare_model_for_training, save_adapter, save_peft_adapter


SPEC = importlib.util.spec_from_file_location(
    "sft_demo", Path(__file__).resolve().parents[1] / "examples/lora_sft_demo.py"
)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture
def tokenizer():
    vocab = {token: index for index, token in enumerate(
        ["<unk>", "<bos>", "<eos>", "<pad>", "<user>", "<assistant>",
         "hello", "world", "answer", "short"]
    )}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>",
        eos_token="<eos>", pad_token="<pad>",
        additional_special_tokens=["<user>", "<assistant>"],
    )


@pytest.fixture
def tiny_model_dir(tmp_path, tokenizer):
    path = tmp_path / "base"
    torch.manual_seed(3407)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=3,
    ))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(autouse=True)
def restore_execution_settings():
    import os

    environment = {key: os.environ.get(key) for key in demo.DETERMINISM_ENV}
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.allow_tf32)
    precision = torch.get_float32_matmul_precision()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32 = cudnn
    torch.set_float32_matmul_precision(precision)
    for key, value in environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def args_for_pairs(*extra):
    return demo.build_parser().parse_args(["train", "--data-format", "prompt-completion", *extra])


def test_completion_mask_changes_only_labels_and_counts_shifted_tokens(tokenizer):
    row = {"prompt": "<bos><user>hello<assistant>", "completion": "answer<eos>"}
    full = demo.tokenize_example(row, tokenizer, args_for_pairs(), 16)
    masked = demo.tokenize_example(row, tokenizer, args_for_pairs("--completion-only-loss"), 16)
    assert full["input_ids"] == masked["input_ids"] == [1, 4, 6, 5, 8, 2]
    assert full["attention_mask"] == masked["attention_mask"]
    assert full["labels"] == [1, 4, 6, 5, 8, 2]
    assert masked["labels"] == [-100, -100, -100, -100, 8, 2]
    assert sum(label != -100 for label in masked["labels"][1:]) == 2
    without_eos = dict(row, completion="answer")
    assert demo.tokenize_example(without_eos, tokenizer, args_for_pairs(), 16) == full
    truncated = demo.tokenize_example(row, tokenizer, args_for_pairs("--completion-only-loss"), 5)
    assert truncated["labels"] == [-100, -100, -100, -100, 8]
    with pytest.raises(ValueError, match="No supervised tokens"):
        demo.tokenize_example(row, tokenizer, args_for_pairs("--completion-only-loss"), 4)


@pytest.mark.parametrize("row", [
    {"prompt": "", "completion": "answer"},
    {"prompt": "hello", "completion": ""},
    {"prompt": ["hello"], "completion": "answer"},
    {"prompt": "hello"},
])
def test_bad_pair_data_is_rejected(row, tokenizer):
    with pytest.raises(ValueError, match="non-empty string"):
        demo.tokenize_example(row, tokenizer, args_for_pairs(), 16)


def test_text_mask_and_causal_empty_examples_are_rejected(tokenizer):
    args = demo.build_parser().parse_args(["train", "--completion-only-loss"])
    with pytest.raises(ValueError, match="requires prompt-completion"):
        demo.tokenize_example({"text": "hello"}, tokenizer, args, 16)
    args = args_for_pairs("--completion-only-loss")
    with pytest.raises(ValueError, match="at least 2"):
        demo.tokenize_example({"prompt": "hello", "completion": "answer"}, tokenizer, args, 1)
    bare = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<eos>": 0}, unk_token="<eos>")),
        unk_token="<eos>", eos_token="<eos>",
    )
    with pytest.raises(ValueError, match="No supervised tokens"):
        demo.tokenize_example({"text": "<eos>"}, bare, demo.build_parser().parse_args(["train"]), 8)


def test_token_boundary_mismatch_is_rejected():
    backend = Tokenizer(models.BPE({"<unk>": 0, "<eos>": 1, "a": 2, "b": 3, "ab": 4}, [("a", "b")]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>")
    for args in (args_for_pairs(), args_for_pairs("--completion-only-loss")):
        with pytest.raises(ValueError, match="prompt token prefix"):
            demo.tokenize_example({"prompt": "a", "completion": "b"}, tokenizer, args, 16)


def test_formatted_generation_passes_one_bos_to_real_model(tokenizer, tiny_model_dir):
    model = LlamaForCausalLM.from_pretrained(tiny_model_dir)
    observed = []
    handle = model.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["input_ids"].clone()), with_kwargs=True,
    )
    try:
        demo.generate_responses(
            model, tokenizer, ["hello"], "<bos><user>{prompt}<assistant>",
            max_new_tokens=1, add_special_tokens=False,
        )
    finally:
        handle.remove()
    assert observed[0].tolist() == [[1, 4, 6, 5]]


def test_legacy_text_matches_real_trl_preprocessing_and_new_collator(tmp_path, tokenizer, tiny_model_dir):
    rows = Dataset.from_list([
        {"text": "hello world"},
        {"text": "<bos><user>hello<assistant>answer<eos>"},
        {"text": "hello world answer short hello world"},
    ])
    model = LlamaForCausalLM.from_pretrained(tiny_model_dir, attn_implementation="eager")
    legacy = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=rows,
        args=SFTConfig(output_dir=str(tmp_path / "legacy"), max_length=6, bf16=False,
                       use_cpu=True, report_to="none", gradient_checkpointing=False),
    )
    args = demo.build_parser().parse_args(["train"])
    prepared = demo.tokenize_dataset(rows, tokenizer, args, 6)
    assert [row["input_ids"] for row in prepared] == [row["input_ids"] for row in legacy.train_dataset]
    # Already formatted text preserves the previous tokenizer's extra BOS.
    assert prepared[1]["input_ids"][:2] == [1, 1]
    actual = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=prepared,
        data_collator=demo.DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        args=SFTConfig(output_dir=str(tmp_path / "prepared"), max_length=6, bf16=False,
                       use_cpu=True, report_to="none", gradient_checkpointing=False,
                       dataset_kwargs={"skip_prepare_dataset": True}),
    )
    batch = actual.data_collator([prepared[0], prepared[1]])
    assert batch["input_ids"][0, :len(prepared[0]["input_ids"])].tolist() == prepared[0]["input_ids"]
    assert batch["labels"][0, len(prepared[0]["labels"]):].eq(-100).all()
    assert batch["labels"][1].tolist() == prepared[1]["labels"]


def test_completion_labels_reach_real_trl_dataloader(tmp_path, tokenizer, tiny_model_dir):
    rows = Dataset.from_list([
        {"prompt": "<bos><user>hello<assistant>", "completion": "answer"},
        {"prompt": "<bos><user>world<assistant>", "completion": "answer short"},
    ])
    batches = []
    for completion_only in (False, True):
        args = args_for_pairs(*(["--completion-only-loss"] if completion_only else []))
        prepared = demo.tokenize_dataset(rows, tokenizer, args, 16)
        trainer = SFTTrainer(
            model=LlamaForCausalLM.from_pretrained(tiny_model_dir),
            processing_class=tokenizer, train_dataset=prepared,
            data_collator=demo.DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
            args=SFTConfig(output_dir=str(tmp_path / str(completion_only)), bf16=False,
                           use_cpu=True, report_to="none", gradient_checkpointing=False,
                           per_device_train_batch_size=2, seed=3407,
                           dataset_kwargs={"skip_prepare_dataset": True}),
        )
        batch = next(iter(trainer.get_train_dataloader()))
        for index, input_ids in enumerate(batch["input_ids"]):
            length = int(batch["attention_mask"][index].sum())
            expected = next(row for row in prepared if row["input_ids"] == input_ids[:length].tolist())
            assert batch["labels"][index, :length].tolist() == expected["labels"]
            assert batch["labels"][index, length:].eq(-100).all()
        batches.append(batch)
    assert torch.equal(batches[0]["input_ids"], batches[1]["input_ids"])
    assert torch.equal(batches[0]["attention_mask"], batches[1]["attention_mask"])
    assert int((batches[1]["labels"][:, 1:] != -100).sum()) == 5
    assert not torch.equal(batches[0]["labels"], batches[1]["labels"])


def test_cpu_train_evaluate_merge_share_execution_and_fp32_checkpoint(tmp_path, tiny_model_dir, tokenizer):
    if torch.cuda.is_available():
        pytest.skip("Run this CPU integration with CUDA_VISIBLE_DEVICES='' to isolate device policy")
    data = tmp_path / "pairs.jsonl"
    rows = [
        {"prompt": "<bos><user>hello<assistant>", "completion": "answer"},
        {"prompt": "<bos><user>world<assistant>", "completion": "answer short"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "run"
    common = ["--model-name", str(tiny_model_dir), "--dtype", "float32", "--adapter-dtype", "float32",
              "--amp", "off", "--full-determinism", "--attn-implementation", "eager"]
    data_args = ["--data-path", str(data), "--data-format", "prompt-completion", "--completion-only-loss"]
    train_args = demo.build_parser().parse_args([
        "train", *common, *data_args, "--output-root", str(output), "--max-steps", "2",
        "--max-seq-length", "16", "--batch-size", "2", "--gradient-accumulation-steps", "1",
        "--target-modules", "q_proj,v_proj", "--lora-r", "2", "--lora-alpha", "4",
        "--save-steps", "2", "--warmup-steps", "0", "--eval-data-path", str(data), "--eval-steps", "1",
        "--no-gradient-checkpointing",
    ])
    demo.run_train(train_args)
    metrics = json.loads((output / "run_metrics.json").read_text())
    assert metrics["global_step"] == 2
    assert metrics["execution"]["parameter_dtypes"]["adapter"].keys() == {"torch.float32"}
    state = json.loads((output / "trainer/checkpoint-2/trainer_state.json").read_text())
    assert len([item for item in state["log_history"] if "eval_loss" in item]) == 2
    optimizer = torch.load(output / "trainer/checkpoint-2/optimizer.pt", weights_only=True)
    assert optimizer["state"]
    for parameter in optimizer["state"].values():
        assert parameter["exp_avg"].dtype == parameter["exp_avg_sq"].dtype == torch.float32
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"hello"}\n')
    eval_args = demo.build_parser().parse_args([
        "evaluate", *common, *data_args, "--output-root", str(output), "--variants", "base,native,peft",
        "--eval-offset", "0", "--eval-samples", "2", "--eval-max-length", "16",
        "--max-new-tokens", "2", "--prompts-file", str(prompts), "--prompt-template", "<bos><user>{prompt}<assistant>",
    ])
    demo.run_evaluate(eval_args)
    evaluation = json.loads((output / "evaluation_results.json").read_text())
    assert evaluation["data_processing"] == metrics["execution"]["data_processing"]
    for variant, result in evaluation["results"].items():
        assert result["language_model"]["tokens"] == 5
        assert math.isfinite(result["language_model"]["loss"])
        for key in ("amp", "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
                    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision", "environment"):
            assert result["execution"][key] == metrics["execution"][key]
        assert set(result["execution"]["attention_implementation"].values()) == {"eager"}
    assert metrics["execution"]["trainer_mixed_precision"] == "no"
    assert metrics["execution"]["environment"] == {**{key: value for key, value in demo.DETERMINISM_ENV.items()},
                                                   "CUDA_VISIBLE_DEVICES": "", "NVIDIA_TF32_OVERRIDE": None}
    assert metrics["execution"]["cuda_matmul_allow_tf32"] is False
    model = demo.load_evaluation_model(eval_args, "native", "cpu", torch.float32)
    samples = [demo.tokenize_example(row, tokenizer, train_args, 16) for row in rows]
    with torch.no_grad():
        losses = []
        for sample in samples:
            batch = demo.DataCollatorForSeq2Seq(tokenizer)([sample])
            losses.append(float(model(**batch).loss))
    expected = (losses[0] * 2 + losses[1] * 3) / 5
    assert evaluation["results"]["native"]["language_model"]["loss"] == pytest.approx(expected, abs=1e-7)
    demo.run_merge(demo.build_parser().parse_args([
        "merge", *common, "--adapter", str(output / "adapter_native"),
        "--output-dir", str(output / "merged_model"), "--verify-prompt", "hello world",
    ]))
    merge = json.loads((output / "merged_model/execution_config.json").read_text())
    assert merge["parameter_dtypes"]["adapter"].keys() == {"torch.float32"}
    train_args.max_steps = 3
    train_args.save_steps = 1
    train_args.resume_from_checkpoint = output / "trainer/checkpoint-2"
    demo.run_train(train_args)
    resumed = json.loads((output / "run_metrics.json").read_text())
    assert resumed["global_step"] == 3
    assert resumed["resume_from_checkpoint"] == str(train_args.resume_from_checkpoint)
    resumed_optimizer = torch.load(output / "trainer/checkpoint-3/optimizer.pt", weights_only=True)
    assert all(int(state["step"]) == 3 for state in resumed_optimizer["state"].values())
    train_args.output_root = tmp_path / "one_epoch"
    train_args.max_steps = -1
    train_args.num_train_epochs = 1.0
    train_args.resume_from_checkpoint = None
    demo.run_train(train_args)
    epoch_metrics = json.loads((train_args.output_root / "run_metrics.json").read_text())
    assert epoch_metrics["global_step"] == 1
    assert epoch_metrics["train_metrics"]["epoch"] == 1.0


def test_auto_reload_preserves_unrounded_fp32_native_and_peft(tmp_path, tiny_model_dir):
    base = LlamaForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.bfloat16)
    config = LoRAConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    model = prepare_model_for_training(base, config, adapter_dtype=torch.float32)
    layer = model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        layer.lora_A.fill_(0.123456789)
        layer.lora_B.fill_(0.234567891)
    save_adapter(model, str(tmp_path / "adapter_native"), config)
    save_peft_adapter(model, str(tmp_path / "adapter_peft"), config)
    args = demo.build_parser().parse_args(["evaluate", "--model-name", str(tiny_model_dir), "--output-root", str(tmp_path)])
    for variant in ("native", "peft"):
        restored = demo.load_evaluation_model(args, variant, "cpu", torch.bfloat16)
        for name, parameter in restored.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert parameter.dtype == torch.float32
                original = layer.lora_A if "lora_A" in name else layer.lora_B
                assert torch.equal(parameter, original)
                assert not torch.equal(parameter, parameter.bfloat16().float())
    native = tmp_path / "adapter_native"
    weights = load_file(native / "adapter_model.safetensors")
    first = next(iter(weights))
    weights[first] = weights[first].bfloat16()
    save_file(weights, native / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="Cannot infer"):
        demo.resolve_adapter_dtype("auto", native)
    assert demo.resolve_adapter_dtype("float32", native) == torch.float32
