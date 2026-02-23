import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union


@dataclass
class TokenizerMeta:
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"


class CharTokenizer:
    def __init__(self, stoi: dict, meta: Optional[TokenizerMeta] = None) -> None:
        self.meta = meta or TokenizerMeta()
        self.stoi = dict(stoi)
        self.itos = [""] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token

        self.pad_id = self.stoi[self.meta.pad_token]
        self.bos_id = self.stoi[self.meta.bos_token]
        self.eos_id = self.stoi[self.meta.eos_token]
        self.unk_id = self.stoi[self.meta.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], skip_special: bool = True) -> str:
        pieces = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        for tid in token_ids:
            if tid < 0 or tid >= len(self.itos):
                continue
            if skip_special and tid in special_ids:
                continue
            pieces.append(self.itos[tid])
        return "".join(pieces)

    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "meta": {
                "pad_token": self.meta.pad_token,
                "bos_token": self.meta.bos_token,
                "eos_token": self.meta.eos_token,
                "unk_token": self.meta.unk_token,
            },
            "stoi": self.stoi,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = TokenizerMeta(**data["meta"])
        return cls(stoi=data["stoi"], meta=meta)

    @classmethod
    def build_from_texts(
        cls, texts: Iterable[str], min_freq: int = 1, meta: Optional[TokenizerMeta] = None
    ) -> "CharTokenizer":
        meta = meta or TokenizerMeta()
        counter = Counter()
        for text in texts:
            counter.update(text)

        vocab = [
            meta.pad_token,
            meta.bos_token,
            meta.eos_token,
            meta.unk_token,
        ]
        frequent_chars = sorted([char for char, freq in counter.items() if freq >= min_freq])
        vocab.extend(frequent_chars)
        stoi = {token: idx for idx, token in enumerate(vocab)}
        return cls(stoi=stoi, meta=meta)
