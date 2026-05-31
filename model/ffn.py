"""Feed-forward block.

Two variants:
  - SwiGLU:  down( silu(gate(x)) * up(x) )   — 3 matrices, used by LLaMA/PaLM
  - GELU:    down( gelu(up(x)) )             — 2 matrices, classic GPT

SwiGLU's hidden dim is shrunk to ~8/3 d (the "2/3 rule") so its parameter count
roughly matches a 4d GELU block. See config.ModelConfig.__post_init__.
"""

import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_swiglu: bool = True):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.gate = nn.Linear(d_model, d_ff, bias=False)
            self.up = nn.Linear(d_model, d_ff, bias=False)
            self.down = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.up = nn.Linear(d_model, d_ff, bias=False)
            self.down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_swiglu:
            return self.down(nn.silu(self.gate(x)) * self.up(x))
        return self.down(nn.gelu(self.up(x)))
