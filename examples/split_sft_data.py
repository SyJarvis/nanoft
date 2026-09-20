"""Split formatted prompt/completion JSONL without sharing prompts across splits.

Uses only the standard library. Records, extra fields, and duplicate rows are
preserved. The complete prompt string defines a group; no normalization or
tokenization is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def split_jsonl(
    source: Path, output_dir: Path, validation_size: int, seed: int = 3407
) -> dict:
    """Write a reproducible split into a new directory and return its manifest."""
    if validation_size < 1:
        raise ValueError("validation_size must be a positive number of rows")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_bytes = source.read_bytes()
    lines = source_bytes.splitlines(keepends=True)
    groups: dict[str, list[int]] = {}
    prompts: dict[str, str] = {}
    for index, line in enumerate(lines):
        location = f"{source}: line {index + 1}"
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{location}: invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a JSON object")
        for field in ("prompt", "completion"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{location}: {field} must be a non-empty string")
        prompt = row["prompt"]
        group_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if group_id in prompts and prompts[group_id] != prompt:
            raise ValueError(f"{location}: SHA-256 collision between different prompts")
        prompts[group_id] = prompt
        groups.setdefault(group_id, []).append(index)

    if len(groups) < 2:
        raise ValueError("At least two distinct prompt groups are required")
    smallest_group = min(len(indices) for indices in groups.values())
    if validation_size > len(lines) - smallest_group:
        raise ValueError(
            f"Cannot reserve {validation_size} validation rows and a non-empty train split "
            f"without breaking prompt groups; maximum is {len(lines) - smallest_group}"
        )

    group_order = sorted(groups)
    random.Random(seed).shuffle(group_order)
    validation_groups = set()
    validation_rows = 0
    for group_id in group_order:
        validation_groups.add(group_id)
        validation_rows += len(groups[group_id])
        if validation_rows >= validation_size:
            break
    if len(validation_groups) == len(groups):
        # A shuffled prefix may consume everything even when a valid split exists.
        # Keeping a smallest group in train still meets the requested row target.
        reserved = min(group_order, key=lambda group_id: len(groups[group_id]))
        validation_groups.remove(reserved)

    train_groups = set(groups) - validation_groups
    assert train_groups and validation_groups
    assert not ({prompts[key] for key in train_groups} & {prompts[key] for key in validation_groups})
    manifest = {
        "source": {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(lines),
            "unique_prompt_groups": len(groups),
        },
        "seed": seed,
        "requested_validation_rows": validation_size,
        "grouping": "SHA-256 of the exact UTF-8 prompt string; no normalization",
        "selection": (
            "Sort group hashes, shuffle with Python random.Random(seed), take whole groups "
            "until the validation row target is met. If this consumes all groups, reserve "
            "a smallest group for train (ties follow shuffled order). Preserve input row "
            "order within each split, including duplicates."
        ),
        "prompt_group_overlap": 0,
        "splits": {},
    }
    # Exclusive creation refuses existing directories, including empty ones.
    # A failed write may leave an incomplete directory; it is never overwritten.
    output_dir.mkdir(parents=True)
    for name, group_ids in (("train", train_groups), ("validation", validation_groups)):
        indices = sorted(index for group_id in group_ids for index in groups[group_id])
        data = b"".join(lines[index] for index in indices)
        filename = f"{name}.jsonl"
        with (output_dir / filename).open("xb") as handle:
            handle.write(data)
        manifest["splits"][name] = {
            "file": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(indices),
            "unique_prompt_groups": len(group_ids),
            "prompt_sha256": sorted(group_ids),
            "source_lines": [index + 1 for index in indices],
        }
    assert manifest["splits"]["validation"]["rows"] >= validation_size
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Formatted prompt/completion JSONL.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; must not exist.")
    parser.add_argument("--validation-size", required=True, type=int, help="Target rows; whole groups may exceed it.")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    try:
        manifest = split_jsonl(args.input, args.output_dir, args.validation_size, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({name: {key: split[key] for key in ("rows", "unique_prompt_groups")}
                      for name, split in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
