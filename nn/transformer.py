from __future__ import annotations

import math
import numpy as np

from tensor.tensor import Tensor
from .embedding import Embedding
from .linear import Linear
from .module import Module
from .norm import LayerNorm


class MultiHeadAttention(Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if causal:
            seq_len = x.shape[1]
            mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float64), k=1) * -1e9
            scores = scores + Tensor(mask.reshape(1, 1, seq_len, seq_len))
        attention = scores.softmax(axis=-1)
        context = attention @ v
        return self.out_proj(self._merge_heads(context))


class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), causal=True)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(block_size, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids) -> Tensor:
        token_ids = np.array(token_ids, dtype=np.int64)
        batch, seq_len = token_ids.shape
        positions = np.tile(np.arange(seq_len, dtype=np.int64), (batch, 1))
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
