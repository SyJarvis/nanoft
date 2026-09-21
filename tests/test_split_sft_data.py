from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "split_sft_data.py"
split_jsonl = runpy.run_path(str(SCRIPT))["split_jsonl"]


@pytest.fixture
def formatted_data(tmp_path):
    rows = [
        {"prompt": "<user>问题 A<assistant>", "completion": "答案一</end>", "metadata": {"source": 1}},
        {"prompt": "<user>问题 B<assistant>", "completion": "答案二</end>", "extra": [1, 2]},
        {"prompt": "<user>问题 A<assistant>", "completion": "不同答案</end>"},
        {"prompt": "<user>问题 C<assistant>", "completion": "答案三</end>"},
        {"prompt": "<user>问题 A<assistant>", "completion": "答案一</end>", "metadata": {"source": 1}},
    ]
    source = tmp_path / "source.jsonl"
    # Keep varied formatting, CRLF, and an unterminated final record observable.
    lines = [json.dumps(row, ensure_ascii=False).encode() + b"\r\n" for row in rows]
    lines[-1] = lines[-1].rstrip(b"\r\n")
    source.write_bytes(b"".join(lines))
    return source, rows, lines


def test_preserves_records_groups_source_order_and_hashes(formatted_data, tmp_path):
    source, rows, lines = formatted_data
    output = tmp_path / "split"
    manifest = split_jsonl(source, output, validation_size=2)
    actual_rows = []
    prompt_sets = []
    for name in ("train", "validation"):
        details = manifest["splits"][name]
        data = (output / details["file"]).read_bytes()
        selected = [json.loads(line) for line in data.splitlines()]
        actual_rows.extend(selected)
        prompt_sets.append({row["prompt"] for row in selected})
        assert selected
        assert data == b"".join(lines[number - 1] for number in details["source_lines"])
        assert details["source_lines"] == sorted(details["source_lines"])
        assert details["rows"] == len(selected)
        assert details["unique_prompt_groups"] == len(prompt_sets[-1])
        assert details["prompt_sha256"] == sorted(hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompt_sets[-1])
        assert details["sha256"] == hashlib.sha256(data).hexdigest()
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert Counter(json.dumps(row, sort_keys=True) for row in actual_rows) == Counter(json.dumps(row, sort_keys=True) for row in rows)
    assert manifest["source"]["rows"] == 5
    assert manifest["source"]["unique_prompt_groups"] == 3
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["prompt_group_overlap"] == 0
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_seed_is_reproducible_and_outputs_change_for_other_seeds(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"prompt": str(i), "completion": "answer"}) + "\n" for i in range(20)))
    first = split_jsonl(source, tmp_path / "first", validation_size=6, seed=3407)
    second = split_jsonl(source, tmp_path / "second", validation_size=6, seed=3407)
    third = split_jsonl(source, tmp_path / "third", validation_size=6, seed=3408)
    assert first == second
    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert first["splits"]["validation"]["source_lines"] != third["splits"]["validation"]["source_lines"]


def test_whitespace_in_nonempty_prompts_is_not_normalized(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"prompt":"q","completion":"a"}\n{"prompt":" q ","completion":"a"}\n')
    manifest = split_jsonl(source, tmp_path / "split", validation_size=1)
    assert manifest["source"]["unique_prompt_groups"] == 2
    assert manifest["splits"]["train"]["rows"] == manifest["splits"]["validation"]["rows"] == 1


def test_whole_groups_can_overshoot_and_keep_train_nonempty(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"prompt": prompt, "completion": "answer"}) + "\n" for prompt in ["q1"] * 3 + ["q2"] * 3))
    manifest = split_jsonl(source, tmp_path / "split", validation_size=2)
    assert manifest["requested_validation_rows"] == 2
    assert manifest["splits"]["validation"]["rows"] == 3
    assert manifest["splits"]["train"]["rows"] == 3


def test_feasible_target_does_not_fail_when_shuffled_prefix_uses_every_group(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"prompt": prompt, "completion": "answer"}) + "\n" for prompt in ["large"] * 3 + ["small"]))
    # Across these seeds both possible group orders are exercised.
    for seed in range(6):
        manifest = split_jsonl(source, tmp_path / f"split-{seed}", validation_size=3, seed=seed)
        assert manifest["splits"]["validation"]["rows"] == 3
        assert manifest["splits"]["train"]["rows"] == 1


@pytest.mark.parametrize("contents, target, message", [
    ("", 1, "two distinct prompt groups"),
    ('{"prompt":"q","completion":"a"}\n{"prompt":"q","completion":"b"}\n', 1, "two distinct prompt groups"),
    ('{"prompt":"q","completion":"a"}\n{"prompt":"r","completion":"b"}\n', 2, "maximum is 1"),
    ('{"prompt":"q","completion":"a"}\n', 0, "positive number"),
    ('{"prompt":"q","completion":"a"}\n', -1, "positive number"),
])
def test_rejects_unsplittable_inputs_without_creating_output(tmp_path, contents, target, message):
    source = tmp_path / "source.jsonl"
    source.write_text(contents)
    output = tmp_path / "split"
    with pytest.raises(ValueError, match=message):
        split_jsonl(source, output, target)
    assert not output.exists()


@pytest.mark.parametrize("bad_row, message", [
    ('{"prompt":"bad"', "invalid UTF-8 JSON"),
    ('[]', "JSON object"),
    ('{"prompt":5,"completion":"a"}', "prompt must be a non-empty string"),
    ('{"prompt":"   ","completion":"a"}', "prompt must be a non-empty string"),
    ('{"completion":"a"}', "prompt must be a non-empty string"),
    ('{"prompt":"q","completion":null}', "completion must be a non-empty string"),
    ('{"prompt":"q","completion":""}', "completion must be a non-empty string"),
    ('', "invalid UTF-8 JSON"),
])
def test_invalid_records_report_physical_line_number(tmp_path, bad_row, message):
    source = tmp_path / "source.jsonl"
    source.write_text('{"prompt":"valid","completion":"answer"}\n' + bad_row + '\n')
    output = tmp_path / "split"
    with pytest.raises(ValueError, match=f"line 2:.*{message}"):
        split_jsonl(source, output, validation_size=1)
    assert not output.exists()


def test_invalid_utf8_reports_line_number(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"prompt":"valid","completion":"answer"}\n\xff\n')
    with pytest.raises(ValueError, match="line 2: invalid UTF-8 JSON"):
        split_jsonl(source, tmp_path / "split", validation_size=1)


@pytest.mark.parametrize("existing_artifact", [None, "train.jsonl", "validation.jsonl", "manifest.json"])
def test_refuses_existing_directory_without_changing_it(formatted_data, tmp_path, existing_artifact):
    source, _, _ = formatted_data
    output = tmp_path / "split"
    output.mkdir()
    if existing_artifact:
        (output / existing_artifact).write_text("keep me\n")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="already exists"):
        split_jsonl(source, output, validation_size=1)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_cli_end_to_end_and_no_overwrite(formatted_data, tmp_path):
    source, _, _ = formatted_data
    output = tmp_path / "cli-split"
    command = [sys.executable, "-I", "-S", str(SCRIPT), "--input", str(source), "--output-dir", str(output), "--validation-size", "2", "--seed", "17"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["train"]["rows"] + summary["validation"]["rows"] == 5
    assert summary["validation"]["rows"] >= 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed"] == 17
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
