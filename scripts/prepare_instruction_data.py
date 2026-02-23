#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

# Make project imports work when script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bruno_core.tokenizer import CharTokenizer


def parse_record(line: str, line_number: int) -> Tuple[str, str]:
    record = json.loads(line)
    user = (
        record.get("user")
        or record.get("Пользователь")
        or record.get("prompt")
        or record.get("instruction")
    )
    bruno = record.get("bruno") or record.get("Bruno") or record.get("assistant") or record.get(
        "output"
    )
    if not user or not bruno:
        raise ValueError(f"Missing user/bruno fields at line {line_number}")
    return str(user).strip(), str(bruno).strip()


def load_records(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        records.append(parse_record(line, idx))
    if not records:
        raise ValueError("Input JSONL does not contain valid examples.")
    return records


def build_example(tokenizer: CharTokenizer, user_text: str, bruno_text: str) -> Tuple[List[int], List[int]]:
    prompt = f"Пользователь: {user_text}\nBruno:"
    answer = f" {bruno_text}"

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)

    full_ids = prompt_ids + answer_ids
    full_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    label_mask = full_mask[1:]
    return input_ids, [token if mask == 1 else -100 for token, mask in zip(labels, label_mask)]


def apply_block_size(
    examples: Sequence[Tuple[List[int], List[int]]],
    block_size: int,
    pad_id: int,
    truncate_mode: str,
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    truncated_count = 0
    too_long = 0

    for input_ids, labels in examples:
        if len(input_ids) > block_size:
            if truncate_mode == "error":
                too_long += 1
                continue
            input_ids = input_ids[:block_size]
            labels = labels[:block_size]
            truncated_count += 1

        if len(input_ids) < block_size:
            pad_len = block_size - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

        input_rows.append(input_ids)
        label_rows.append(labels)

    return input_rows, label_rows, truncated_count, too_long


def save_payload(
    out_path: str,
    input_rows: Sequence[Sequence[int]],
    label_rows: Sequence[Sequence[int]],
    block_size: int,
    truncated_count: int,
    split_name: str,
) -> None:
    payload = {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "block_size": block_size,
        "num_examples": len(input_rows),
        "truncated_examples": truncated_count,
        "split": split_name,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(
        f"Saved {len(input_rows)} examples to {path} "
        f"(split={split_name}, block_size={block_size}, truncated={truncated_count})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare user/bruno instruction dataset.")
    parser.add_argument("--input", required=True, help="Path to input JSONL.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--out", required=True, help="Path to output train .pt file.")
    parser.add_argument("--out-val", default=None, help="Optional output val .pt file.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--block-size", type=int, default=512, help="Max sequence length.")
    parser.add_argument(
        "--truncate-mode",
        choices=["error", "truncate"],
        default="error",
        help="Use 'error' to fail instead of truncating long examples.",
    )
    parser.add_argument(
        "--auto-block-size",
        action="store_true",
        help="Auto-increase block size to longest example (never below --block-size).",
    )
    args = parser.parse_args()

    if args.out_val and (args.val_ratio <= 0 or args.val_ratio >= 1):
        raise ValueError("--val-ratio must be between 0 and 1 when --out-val is provided.")

    tokenizer = CharTokenizer.load(args.tokenizer)
    records = load_records(args.input)
    built_examples = [build_example(tokenizer, user_text, bruno_text) for user_text, bruno_text in records]

    if not built_examples:
        raise ValueError("No valid examples were built from the input dataset.")

    lengths = [len(input_ids) for input_ids, _ in built_examples]
    max_len = max(lengths)
    block_size = max(args.block_size, max_len) if args.auto_block_size else args.block_size

    if args.auto_block_size and block_size != args.block_size:
        print(f"Auto block size enabled: increased block_size from {args.block_size} to {block_size}")

    if args.out_val:
        shuffled = list(built_examples)
        random.Random(args.seed).shuffle(shuffled)
        val_count = int(len(shuffled) * args.val_ratio)
        if val_count <= 0:
            val_count = 1
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1
        val_examples = shuffled[:val_count]
        train_examples = shuffled[val_count:]
    else:
        train_examples = built_examples
        val_examples = []

    train_inputs, train_labels, train_truncated, train_too_long = apply_block_size(
        examples=train_examples,
        block_size=block_size,
        pad_id=tokenizer.pad_id,
        truncate_mode=args.truncate_mode,
    )
    if not train_inputs:
        raise ValueError("Train split has no usable examples after preprocessing.")

    if args.truncate_mode == "error" and train_too_long > 0:
        raise ValueError(
            f"Train split has {train_too_long} examples longer than block_size={block_size}. "
            f"Max length is {max_len}. Increase --block-size or use --auto-block-size."
        )

    save_payload(
        out_path=args.out,
        input_rows=train_inputs,
        label_rows=train_labels,
        block_size=block_size,
        truncated_count=train_truncated,
        split_name="train",
    )

    if args.out_val:
        val_inputs, val_labels, val_truncated, val_too_long = apply_block_size(
            examples=val_examples,
            block_size=block_size,
            pad_id=tokenizer.pad_id,
            truncate_mode=args.truncate_mode,
        )
        if not val_inputs:
            raise ValueError("Validation split has no usable examples after preprocessing.")
        if args.truncate_mode == "error" and val_too_long > 0:
            raise ValueError(
                f"Val split has {val_too_long} examples longer than block_size={block_size}. "
                f"Max length is {max_len}. Increase --block-size or use --auto-block-size."
            )

        save_payload(
            out_path=args.out_val,
            input_rows=val_inputs,
            label_rows=val_labels,
            block_size=block_size,
            truncated_count=val_truncated,
            split_name="val",
        )

        summary: Dict[str, int] = {
            "train_examples": len(train_inputs),
            "val_examples": len(val_inputs),
            "block_size": block_size,
            "max_raw_example_len": max_len,
            "train_truncated": train_truncated,
            "val_truncated": val_truncated,
        }
        print("Split summary:", json.dumps(summary, ensure_ascii=False))
    else:
        summary = {
            "train_examples": len(train_inputs),
            "block_size": block_size,
            "max_raw_example_len": max_len,
            "train_truncated": train_truncated,
        }
        print("Summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
