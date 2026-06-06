from __future__ import annotations

import numpy as np

from .dataset import Dataset


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int) -> None:
        self.text = text
        self.block_size = block_size
        self.vocab = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.vocab)}
        self.itos = {index: char for char, index in self.stoi.items()}
        self.encoded = np.array([self.stoi[char] for char in text], dtype=np.int64)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return max(0, len(self.encoded) - self.block_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.encoded[index : index + self.block_size]
        y = self.encoded[index + 1 : index + self.block_size + 1]
        return x, y

    def decode(self, token_ids: list[int] | np.ndarray) -> str:
        return "".join(self.itos[int(token_id)] for token_id in token_ids)
