"""Multi-head attention with optional grouped-query attention (GQA).

Explicit Q, K, V projections — no fused QKV — so every shape is legible.

GQA: n_kv_heads < n_heads means K/V are projected to fewer heads and broadcast
across query-head groups. This shrinks the KV cache (the memory-bandwidth
bottleneck at inference) without much quality loss. Set n_kv_heads == n_heads
for standard MHA.

Causal masking + scaled dot product use MLX's fused
`mx.fast.scaled_dot_product_attention`, which is the flash-style kernel: it
avoids materialising the full (T, T) score matrix in memory.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .rope import RoPE


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.head_dim, theta=rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)  # (B, T, d)
        k = self.k_proj(x)  # (B, T, kv_dim)
        v = self.v_proj(x)

        # split heads -> (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # rotary embeddings on Q and K (offset handles KV-cache decoding)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # fused flash-style attention; handles GQA broadcast internally
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        # merge heads -> (B, T, d)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)
