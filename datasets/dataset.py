from __future__ import annotations

import math
import random
from collections.abc import Iterator

import numpy as np


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            xs, ys = zip(*(self.dataset[index] for index in batch_indices))
            yield np.stack(xs), np.stack(ys)
