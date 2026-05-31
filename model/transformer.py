"""Full decoder-only transformer.

  tokens -> embedding -> N x TransformerBlock -> final norm -> LM head -> logits

Pre-norm residual blocks (norm before the sublayer), RMSNorm, RoPE, SwiGLU —
the modern LLaMA-style recipe. Embedding/LM-head weights are tied by default.
"""

import mlx.core as mx
import mlx.nn as nn

from config import ModelConfig
from .attention import Attention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                              rope_theta=cfg.rope_theta)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, use_swiglu=cfg.use_swiglu)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, idx: mx.array, cache=None) -> mx.array:
        # idx: (B, T) int tokens
        B, T = idx.shape
        x = self.tok_embed(idx)

        if cache is None:
            cache = [None] * len(self.blocks)
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T) if T > 1 else None
        else:
            mask = None  # single-token decode step needs no mask

        for block, c in zip(self.blocks, cache):
            x = block(x, mask=mask, cache=c)

        x = self.final_norm(x)
        if self.cfg.tie_embeddings:
            return self.tok_embed.as_linear(x)  # weight-tied projection
        return self.lm_head(x)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        logits = logits.reshape(-1, self.cfg.vocab_size)
        targets = targets.reshape(-1)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")
