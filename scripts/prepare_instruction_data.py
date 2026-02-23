#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

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


def build_example(tokenizer: CharTokenizer, user_text: str, bruno_text: str) -> Tuple[list, list]:
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


def pad_or_truncate(
    input_ids: list[int], labels: list[int], block_size: int, pad_id: int
) -> Tuple[list[int], list[int], bool]:
    truncated = False
    if len(input_ids) > block_size:
        input_ids = input_ids[:block_size]
        labels = labels[:block_size]
        truncated = True

    if len(input_ids) < block_size:
        pad_len = block_size - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return input_ids, labels, truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare user/bruno instruction dataset.")
    parser.add_argument("--input", required=True, help="Path to input JSONL.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--out", required=True, help="Path to output .pt file.")
    parser.add_argument("--block-size", type=int, default=256, help="Max sequence length.")
    args = parser.parse_args()

    tokenizer = CharTokenizer.load(args.tokenizer)
    input_rows = []
    label_rows = []
    truncated_count = 0

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        user_text, bruno_text = parse_record(line, idx)
        input_ids, labels = build_example(tokenizer, user_text, bruno_text)

        if len(input_ids) < 1:
            continue

        input_ids, labels, truncated = pad_or_truncate(
            input_ids=input_ids,
            labels=labels,
            block_size=args.block_size,
            pad_id=tokenizer.pad_id,
        )
        if truncated:
            truncated_count += 1

        input_rows.append(input_ids)
        label_rows.append(labels)

    if not input_rows:
        raise ValueError("No valid examples were built from the input dataset.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "block_size": args.block_size,
        "num_examples": len(input_rows),
        "truncated_examples": truncated_count,
    }
    torch.save(payload, out_path)

    print(f"Saved {len(input_rows)} examples to {out_path}")
    if truncated_count > 0:
        print(f"Truncated examples: {truncated_count}")


if __name__ == "__main__":
    main()
