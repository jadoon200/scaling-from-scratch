"""Rotary positional embeddings (RoPE).

Rotates pairs of dimensions in Q and K by a position-dependent angle. The dot
product Q·K then depends only on the *relative* position (i - j), which is what
gives RoPE its length-extrapolation behaviour.

MLX ships a fused `mx.fast.rope`; we use it for speed but the math below is
exactly the interleaved-pair rotation it implements.
"""

import mlx.core as mx
import mlx.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.head_dim = head_dim
        self.theta = theta

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: (batch, n_heads, seq_len, head_dim)
        return mx.fast.rope(
            x,
            dims=self.head_dim,
            traditional=False,
            base=self.theta,
            scale=1.0,
            offset=offset,
        )
