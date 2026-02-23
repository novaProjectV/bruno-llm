#!/usr/bin/env python3
import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make project imports work when script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bruno_core.model import GPT, GPTConfig


class TensorDataset(Dataset):
    def __init__(self, payload: dict) -> None:
        self.input_ids = payload["input_ids"]
        self.labels = payload["labels"]
        if self.input_ids.size(0) != self.labels.size(0):
            raise ValueError("input_ids and labels must have the same number of examples")

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


def _pick_first(payload: dict, keys: list[str]):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[GPT, dict]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = _pick_first(raw, ["model_config", "model_args", "config"])
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing model config (model_config/model_args/config).")

    state_dict = _pick_first(raw, ["model_state", "model", "state_dict"])
    if state_dict is None:
        raise ValueError("Checkpoint is missing model state (model_state/model/state_dict).")

    cleaned_state = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state[cleaned_key] = value

    config = GPTConfig.from_dict(cfg_dict)
    model = GPT(config)
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint load ({len(missing)}).")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint load ({len(unexpected)}).")

    model.to(device)
    return model, raw


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def format_epoch_metrics(epoch: int, train_loss: float, val_loss: Optional[float]) -> str:
    base = f"Epoch {epoch}: train_loss={train_loss:.4f} train_ppl={math.exp(train_loss):.2f}"
    if val_loss is not None:
        base += f" | val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction tuning for Bruno assistant.")
    parser.add_argument("--base-checkpoint", required=True, help="Path to Bruno Core 1 checkpoint.")
    parser.add_argument("--train-data", required=True, help="Path to prepared .pt train data.")
    parser.add_argument("--val-data", default=None, help="Optional prepared .pt validation data.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path for metadata.")
    parser.add_argument("--out-dir", required=True, help="Output directory for assistant checkpoint.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--prototype-name",
        default="Bruno Prototype 0.2",
        help="Name/version saved into checkpoint metadata.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, base_raw = load_model_from_checkpoint(args.base_checkpoint, device)

    train_payload = torch.load(args.train_data, map_location="cpu")
    train_dataset = TensorDataset(train_payload)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = None
    if args.val_data:
        val_payload = torch.load(args.val_data, map_location="cpu")
        val_dataset = TensorDataset(val_payload)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    train_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())

            if global_step % args.log_every == 0:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"train_loss={sum(epoch_losses) / len(epoch_losses):.4f}"
                )

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": math.exp(train_loss),
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss) if val_loss is not None else None,
        }
        train_history.append(history_row)
        print(format_epoch_metrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "parent_checkpoint": str(args.base_checkpoint),
            "tokenizer_path": args.tokenizer or base_raw.get("tokenizer_path"),
            "prototype_name": args.prototype_name,
            "epoch": epoch,
            "global_step": global_step,
            "train_history": train_history,
            "train_data": str(args.train_data),
            "val_data": str(args.val_data) if args.val_data else None,
        }

        last_path = out_dir / "checkpoint_last.pt"
        torch.save(checkpoint, last_path)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "checkpoint_best.pt")
            print(f"Saved new best checkpoint: val_loss={val_loss:.4f}")

    (out_dir / "train_history.json").write_text(
        json.dumps(train_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train_history) + "\n",
        encoding="utf-8",
    )
    print(f"Training complete. Last checkpoint: {out_dir / 'checkpoint_last.pt'}")


if __name__ == "__main__":
    main()
