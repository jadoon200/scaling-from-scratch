"""Model and training configuration.

All hyperparameters live here. The dataclass also computes derived quantities
(parameter count, FLOPs per token) so they can be logged without instantiating
the model — useful for planning scaling-law experiments before training.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Apple Silicon peak specs. Override per chip; defaults are M3 Pro (18-core GPU).
PEAK_FLOPS_FP16 = 6.4e12   # FLOP/s
PEAK_BW_BYTES = 150e9      # bytes/s
RIDGE_POINT = PEAK_FLOPS_FP16 / PEAK_BW_BYTES  # ~43 FLOP/byte


@dataclass
class ModelConfig:
    # --- architecture ---
    vocab_size: int = 50304       # GPT-2 BPE rounded up to a multiple of 64
    d_model: int = 256            # residual stream width
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None  # None => MHA; set < n_heads for GQA
    d_ff: int | None = None        # None => 4 * d_model (or SwiGLU-adjusted)
    seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True    # share token embedding with LM head
    use_swiglu: bool = True        # SwiGLU FFN vs GELU

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide evenly into n_heads"
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        if self.d_ff is None:
            # SwiGLU has 3 matrices instead of 2, so shrink the hidden dim to
            # roughly match the parameter count of a 4*d GELU FFN (the 2/3 rule).
            self.d_ff = int(8 / 3 * self.d_model) if self.use_swiglu else 4 * self.d_model
            self.d_ff = (self.d_ff + 63) // 64 * 64  # round to multiple of 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params_embedding(self) -> int:
        """Token embedding params (+ untied LM head if not tied)."""
        embed = self.vocab_size * self.d_model
        if not self.tie_embeddings:
            embed += self.vocab_size * self.d_model
        return embed

    @property
    def n_params_non_embedding(self) -> int:
        """Transformer-block params (ignores norm/bias terms, which are tiny).

        This is the quantity scaling laws (Chinchilla, Kaplan) actually track,
        because embedding params don't contribute FLOPs the same way and wash
        out as the model grows.
        """
        d = self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        # attention: q (d*d) + k,v (d*kv_dim each) + out (d*d)
        attn = d * d + 2 * d * kv_dim + d * d
        ffn = (3 if self.use_swiglu else 2) * d * self.d_ff
        return self.n_layers * (attn + ffn)

    @property
    def n_params(self) -> int:
        """Total parameter count."""
        return self.n_params_embedding + self.n_params_non_embedding

    def flops_per_token(self) -> float:
        """Forward-pass FLOPs per token (Chinchilla-style 2*params + attention)."""
        d, L, T = self.d_model, self.n_layers, self.seq_len
        # 2 * params accounts for all matmuls (multiply + add)
        param_flops = 2 * self.n_params
        # attention scores + values: 2 * (2 * T * d) per layer per token
        attn_flops = L * 4 * T * d
        return param_flops + attn_flops

    def flops_per_step(self, tokens_per_step: int) -> float:
        """Training-step FLOPs ~= 6 * N * tokens (fwd + bwd)."""
        return 6 * self.n_params * tokens_per_step


@dataclass
class TrainConfig:
    # --- optimization ---
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 5000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- logging / checkpointing ---
    log_every: int = 10
    eval_every: int = 250
    eval_steps: int = 50
    ckpt_every: int = 1000
    ckpt_dir: str = "checkpoints"
    run_name: str = "run"

    # --- data ---
    data_dir: str = "data/fineweb"
    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        # filled in against ModelConfig.seq_len at runtime
        return self.batch_size * self.grad_accum_steps


# Preset configs for scaling-law sweeps, named by NON-EMBEDDING params (the
# quantity scaling laws track). Train each to its Chinchilla-optimal token
# budget (~20 tokens per non-embedding param) and record val loss.
SCALING_PRESETS: dict[str, ModelConfig] = {
    "1M":   ModelConfig(d_model=192, n_layers=4, n_heads=6, seq_len=256),
    "3M":   ModelConfig(d_model=288, n_layers=6, n_heads=6, seq_len=512),
    "10M":  ModelConfig(d_model=448, n_layers=8, n_heads=8, seq_len=512),
    "30M":  ModelConfig(d_model=640, n_layers=10, n_heads=10, seq_len=512),
    "100M": ModelConfig(d_model=896, n_layers=14, n_heads=14, seq_len=512),
}


if __name__ == "__main__":
    print(f"vocab_size={ModelConfig().vocab_size}, ridge point ≈ {RIDGE_POINT:.0f} FLOP/byte\n")
    print(f"{'name':>5} {'non-embed':>10} {'total':>9} {'embed%':>7}  arch")
    for name, cfg in SCALING_PRESETS.items():
        ne = cfg.n_params_non_embedding
        tot = cfg.n_params
        embed_pct = 100 * cfg.n_params_embedding / tot
        chinchilla = 20 * ne
        print(f"{name:>5} {ne/1e6:8.2f}M {tot/1e6:7.2f}M {embed_pct:6.1f}%  "
              f"d={cfg.d_model:>4} L={cfg.n_layers:>2} h={cfg.n_heads:>2} "
              f"d_ff={cfg.d_ff:>4}  D≈{chinchilla/1e6:.0f}M tok")
