from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import CharDataset, DataLoader
from nn import TinyGPT, cross_entropy
from optim import Adam
from tensor.tensor import Tensor


def main() -> None:
    np.random.seed(0)
    text = (
        "hello how are you\n"
        "hello there friend\n"
        "tiny gpt learns small text\n"
        "sequence models are fun\n"
    ) * 20
    block_size = 12
    dataset = CharDataset(text, block_size=block_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TinyGPT(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
    )
    optimizer = Adam(model.parameters(), lr=0.01)

    for step, (x_batch, y_batch) in enumerate(loader, start=1):
        logits = model(x_batch)
        loss = cross_entropy(
            logits.reshape(-1, dataset.vocab_size),
            y_batch.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(f"step={step:03d} loss={loss.item():.6f}")
        if step >= 30:
            break

    sample = np.array([dataset.stoi[c] for c in "hello how ar"], dtype=np.int64).reshape(1, -1)
    logits = model(sample)
    next_token = int(np.argmax(logits.data[0, -1]))
    print("prompt:", dataset.decode(sample[0]))
    print("next token guess:", repr(dataset.decode([next_token])))


if __name__ == "__main__":
    main()
