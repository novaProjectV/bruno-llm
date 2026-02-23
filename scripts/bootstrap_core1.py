#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Make project imports work when script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bruno_core.model import GPT, GPTConfig
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


def load_instruction_dialogs(path: str) -> List[Tuple[str, str]]:
    dialogs: List[Tuple[str, str]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        dialogs.append(parse_record(line, idx))
    if not dialogs:
        raise ValueError("No dialogs found in instruction JSONL.")
    return dialogs


def build_corpus(dialogs: List[Tuple[str, str]], repeat: int = 200) -> str:
    # Repeat dialog templates to create a minimally-sized corpus for first pretraining pass.
    blocks = []
    for _ in range(max(1, repeat)):
        random.shuffle(dialogs)
        for user_text, bruno_text in dialogs:
            blocks.append(f"Пользователь: {user_text}\nBruno: {bruno_text}\n\n")
    return "".join(blocks)


def sample_batch(
    token_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = token_ids.size(0) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x_rows = [token_ids[i : i + block_size] for i in starts]
    y_rows = [token_ids[i + 1 : i + 1 + block_size] for i in starts]
    x = torch.stack(x_rows).to(device)
    y = torch.stack(y_rows).to(device)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Bruno Core 1 artifacts from instruction JSONL."
    )
    parser.add_argument("--input", required=True, help="Instruction JSONL path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for core artifacts.")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--corpus-repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"
    checkpoint_path = out_dir / "bruno_core1.pt"

    dialogs = load_instruction_dialogs(args.input)
    corpus = build_corpus(dialogs, repeat=args.corpus_repeat)

    tokenizer = CharTokenizer.build_from_texts([corpus], min_freq=1)
    tokenizer.save(tokenizer_path)

    token_ids = tokenizer.encode(corpus, add_bos=True, add_eos=True)
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    if token_tensor.size(0) <= args.block_size + 1:
        raise ValueError(
            f"Corpus too short ({token_tensor.size(0)} tokens). "
            f"Increase --corpus-repeat or reduce --block-size."
        )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    model.train()

    losses: List[float] = []
    for step in range(1, args.steps + 1):
        x, y = sample_batch(
            token_ids=token_tensor,
            block_size=args.block_size,
            batch_size=args.batch_size,
            device=device,
        )
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 25 == 0 or step == args.steps:
            avg = sum(losses[-25:]) / max(len(losses[-25:]), 1)
            print(f"step={step}/{args.steps} loss={avg:.4f} ppl={math.exp(avg):.2f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": asdict(config),
        "tokenizer_path": str(tokenizer_path),
        "core_stage": "Bruno Core 1",
        "source_instruction_data": str(args.input),
        "train_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    torch.save(checkpoint, checkpoint_path)

    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
