from __future__ import annotations

import unittest

import numpy as np

from nn.conv import Conv2D
from nn.embedding import Embedding
from nn.norm import LayerNorm
from nn.pooling import MaxPool2D
from nn.transformer import MultiHeadAttention, TransformerBlock, TinyGPT
from tensor.tensor import Tensor
from nn.linear import Linear
from nn.losses import mse_loss


class TensorAutogradTests(unittest.TestCase):
    def test_scalar_backward(self) -> None:
        x = Tensor(2.0, requires_grad=True)
        y = x * x
        y.backward()
        self.assertAlmostEqual(float(x.grad), 4.0, places=5)

    def test_matmul_backward(self) -> None:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w = Tensor([[0.5], [0.8]], requires_grad=True)
        loss = (x @ w).mean()
        loss.backward()
        np.testing.assert_allclose(w.grad, np.array([[2.0], [3.0]]), atol=1e-6)

    def test_linear_learns_simple_mapping(self) -> None:
        np.random.seed(0)
        x = Tensor(np.array([[0.0], [1.0], [2.0], [3.0]]))
        y = Tensor(np.array([[1.0], [3.0], [5.0], [7.0]]))
        layer = Linear(1, 1)

        for _ in range(400):
            pred = layer(x)
            loss = mse_loss(pred, y)
            layer.zero_grad()
            loss.backward()
            for param in layer.parameters():
                param.data = param.data - 0.05 * param.grad

        final_loss = mse_loss(layer(x), y).item()
        self.assertLess(final_loss, 1e-3)

    def test_conv2d_output_shape_and_backward(self) -> None:
        np.random.seed(0)
        x = Tensor(np.random.randn(2, 1, 5, 5), requires_grad=True)
        conv = Conv2D(1, 2, kernel_size=3, stride=1, padding=0)
        out = conv(x)
        self.assertEqual(out.shape, (2, 2, 3, 3))
        loss = out.mean()
        loss.backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)

    def test_maxpool2d_output_shape_and_backward(self) -> None:
        x = Tensor(
            np.array([[[[1.0, 3.0, 2.0, 1.0], [4.0, 6.0, 5.0, 2.0], [1.0, 2.0, 9.0, 1.0], [0.0, 1.0, 3.0, 8.0]]]]),
            requires_grad=True,
        )
        pool = MaxPool2D(kernel_size=2, stride=2)
        out = pool(x)
        np.testing.assert_allclose(out.data, np.array([[[[6.0, 5.0], [2.0, 9.0]]]]))
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertAlmostEqual(float(x.grad.sum()), 4.0, places=5)

    def test_embedding_backward(self) -> None:
        embedding = Embedding(8, 4)
        token_ids = np.array([[1, 2, 1]])
        out = embedding(token_ids)
        loss = out.mean()
        loss.backward()
        self.assertEqual(embedding.weight.grad.shape, embedding.weight.shape)
        self.assertGreater(np.abs(embedding.weight.grad[1]).sum(), 0)

    def test_layernorm_shape(self) -> None:
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        ln = LayerNorm(4)
        out = ln(x)
        self.assertEqual(out.shape, x.shape)
        out.mean().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_multihead_attention_shape(self) -> None:
        x = Tensor(np.random.randn(2, 5, 8), requires_grad=True)
        attn = MultiHeadAttention(d_model=8, num_heads=2)
        out = attn(x)
        self.assertEqual(out.shape, (2, 5, 8))
        out.mean().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_transformer_block_shape(self) -> None:
        x = Tensor(np.random.randn(2, 6, 8), requires_grad=True)
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=16)
        out = block(x)
        self.assertEqual(out.shape, (2, 6, 8))

    def test_tiny_gpt_logits_shape(self) -> None:
        model = TinyGPT(vocab_size=20, block_size=8, d_model=16, num_heads=4, num_layers=1, d_ff=32)
        tokens = np.random.randint(0, 20, size=(3, 8))
        logits = model(tokens)
        self.assertEqual(logits.shape, (3, 8, 20))


if __name__ == "__main__":
    unittest.main()
