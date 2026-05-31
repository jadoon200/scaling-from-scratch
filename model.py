"""Compact decoder-only transformer — the quantization eval target.

Single-file, deliberately minimal: RMSNorm pre-norm blocks, RoPE, SwiGLU FFN,
tied embedding/LM head. The point isn't novel architecture (that's project #1);
it's having a real model whose Linear layers we can swap to QuantizedLinear and
measure the quality/speed/memory tradeoff.

`quantize_model(model, bits, group_size)` walks the module tree and replaces
every nn.Linear with a QuantizedLinear in place.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from config import ModelConfig
from quant.qlinear import QuantizedLinear


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, base=cfg.rope_theta)

        d_ff = (int(8 / 3 * cfg.d_model) + 63) // 64 * 64
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.gate = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, cfg.d_model, bias=False)

    def __call__(self, x, mask):
        B, T, _ = x.shape
        h = self.attn_norm(x)
        q = self.q(h).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        x = x + self.o(o)
        h = self.ffn_norm(x)
        x = x + self.down(nn.silu(self.gate(h)) * self.up(h))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_embed(idx)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T) if T > 1 else None
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.final_norm(x)
        return self.tok_embed.as_linear(x)  # tied LM head

    def loss(self, idx, targets):
        logits = self(idx).reshape(-1, self.cfg.vocab_size)
        return nn.losses.cross_entropy(logits, targets.reshape(-1), reduction="mean")


def quantize_model(model: nn.Module, bits: int = 4, group_size: int = 64) -> int:
    """Replace every nn.Linear in the tree with a QuantizedLinear, in place.

    Returns the number of layers quantized. The embedding (and its tied LM head)
    is left in fp — embeddings are looked up, not matmul-streamed, so quantizing
    them buys little and hurts quality most.
    """
    n = 0

    def convert(module: nn.Module):
        nonlocal n
        for name, child in list(module.children().items()):
            if isinstance(child, list):
                for i, c in enumerate(child):
                    if isinstance(c, nn.Linear):
                        child[i] = QuantizedLinear.from_linear(c, bits, group_size)
                        n += 1
                    else:
                        convert(c)
            elif isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear.from_linear(child, bits, group_size))
                n += 1
            elif isinstance(child, nn.Module):
                convert(child)

    convert(model)
    return n


if __name__ == "__main__":
    from mlx.utils import tree_flatten
    cfg = ModelConfig(d_model=512, n_layers=8, n_heads=8, seq_len=256)
    m = Transformer(cfg); mx.eval(m.parameters())
    n_params = sum(p.size for _, p in tree_flatten(m.parameters()))
    x = mx.random.randint(0, cfg.vocab_size, (2, 64))
    y = m(x); mx.eval(y)
    print(f"params: {n_params/1e6:.1f}M | logits: {y.shape}")
    n = quantize_model(m, bits=4, group_size=64)
    yq = m(x); mx.eval(yq)
    print(f"quantized {n} Linear layers | logits after quant: {yq.shape}")
