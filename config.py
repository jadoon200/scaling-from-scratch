"""Model and quantization configuration."""

from __future__ import annotations

from dataclasses import dataclass

# Apple M3 Pro peak specs (18-core GPU)
PEAK_FLOPS_FP16 = 6.4e12   # FLOP/s
PEAK_BW_BYTES = 150e9      # bytes/s
RIDGE_POINT = PEAK_FLOPS_FP16 / PEAK_BW_BYTES  # ~43 FLOP/byte


@dataclass
class ModelConfig:
    """Compact transformer used as the quantization eval target."""
    vocab_size: int = 50304
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class QuantConfig:
    bits: int = 4          # 8 or 4
    group_size: int = 64   # weights per (scale, zero) group; smaller = finer
    symmetric: bool = False

    def effective_bits(self) -> float:
        """Bits per weight including per-group scale + zero (fp16 each)."""
        overhead_bits = (16 + 16) / self.group_size  # scale + zero, 16 bits each
        return self.bits + overhead_bits


if __name__ == "__main__":
    print(f"M3 Pro ridge point ≈ {RIDGE_POINT:.0f} FLOP/byte\n")
    for bits in (8, 4):
        for g in (32, 64, 128):
            qc = QuantConfig(bits=bits, group_size=g)
            print(f"INT{bits} group={g:>3}: {qc.effective_bits():.2f} effective "
                  f"bits/weight  ({16/qc.effective_bits():.2f}x vs fp16)")
