#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

# Make project imports work when script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bruno_core.model import GPT, GPTConfig
from bruno_core.tokenizer import CharTokenizer


def _pick_first(payload: dict, keys: list[str]):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    raw = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = _pick_first(raw, ["model_config", "model_args", "config"])
    state_dict = _pick_first(raw, ["model_state", "model", "state_dict"])
    if cfg_dict is None or state_dict is None:
        raise ValueError("Checkpoint must contain config and state dict.")

    config = GPTConfig.from_dict(cfg_dict)
    model = GPT(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_prompt(history: list[tuple[str, str]], user_message: str, max_turns: int) -> str:
    chunks = []
    for user_text, bruno_text in history[-max_turns:]:
        chunks.append(f"Пользователь: {user_text}\nBruno: {bruno_text}")
    chunks.append(f"Пользователь: {user_message}\nBruno:")
    return "\n\n".join(chunks)


def extract_assistant_text(raw: str) -> str:
    stop_markers = ["\nПользователь:", "\nUser:"]
    answer = raw
    for marker in stop_markers:
        pos = answer.find(marker)
        if pos != -1:
            answer = answer[:pos]
    return answer.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Bruno Assistant checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-turns", type=int, default=4)
    args = parser.parse_args()

    if args.top_p <= 0 or args.top_p > 1:
        raise ValueError("--top-p must be in the range (0, 1].")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0.")

    tokenizer = CharTokenizer.load(args.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    history: list[tuple[str, str]] = []
    print("Bruno chat started. Type 'exit' to stop.")
    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        prompt = build_prompt(history, user_message, max_turns=args.max_turns)
        input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_id=tokenizer.eos_id,
            )

        generated_ids = output[0].tolist()[len(input_ids) :]
        answer_raw = tokenizer.decode(generated_ids, skip_special=True)
        answer = extract_assistant_text(answer_raw)
        if not answer:
            answer = "Извини, я не смог сформулировать ответ. Попробуй перефразировать вопрос."

        print(f"Bruno: {answer}")
        history.append((user_message, answer))


if __name__ == "__main__":
    main()
