"""Roofline profiler for Apple Silicon.

Measures achieved FLOP/s and memory bandwidth for forward and forward+backward
passes across a grid of (batch_size, seq_len), then places each point on the
roofline:

    attainable FLOP/s = min(peak_FLOPs, peak_BW * arithmetic_intensity)

Arithmetic intensity (FLOP/byte) is what determines which roof you hit. Below
the ridge point (~43 FLOP/byte on M3 Pro) you're memory-bound; above it,
compute-bound. Small models with short sequences sit on the memory roof; you
climb toward the compute roof by increasing batch size and sequence length.

Run:
    python profile.py                      # default config, sweep batch/seq
    python profile.py --preset 30M
    python profile.py --backward           # include the backward pass
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from config import ModelConfig, SCALING_PRESETS, PEAK_FLOPS_FP16, PEAK_BW_BYTES, RIDGE_POINT
from model import Transformer


def time_fn(fn, iters: int = 20, warmup: int = 3) -> float:
    """Median-ish wall time per call, with mx.eval barriers."""
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def bytes_moved(n_params: int, acts: int, backward: bool) -> int:
    """Rough bytes streamed through memory in fp16 (2 bytes/elt).

    Weights are read once per pass; activations are written and re-read. This is
    a lower bound (ignores cache reuse) but captures the right scaling.
    """
    w = n_params * 2
    a = acts * 2
    return (w + a) * (3 if backward else 1)  # bwd ~ 2x fwd traffic + weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(SCALING_PRESETS), default=None)
    ap.add_argument("--backward", action="store_true",
                    help="profile forward+backward instead of forward only")
    args = ap.parse_args()

    cfg = SCALING_PRESETS[args.preset] if args.preset else ModelConfig()
    model = Transformer(cfg)
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))

    print(f"model: {n_params/1e6:.2f}M params  d={cfg.d_model} L={cfg.n_layers}")
    print(f"chip:  {PEAK_FLOPS_FP16/1e12:.1f} TFLOP/s, {PEAK_BW_BYTES/1e9:.0f} GB/s, "
          f"ridge {RIDGE_POINT:.0f} FLOP/byte")
    print(f"mode:  {'forward+backward' if args.backward else 'forward'}\n")

    grid = [(1, 128), (1, 512), (8, 512), (32, 512), (32, 1024)]

    hdr = f"{'batch':>5} {'seq':>5} {'ms':>8} {'TFLOP/s':>8} {'GB/s':>7} {'intens':>7} {'MFU':>6} {'bound':>8}"
    print(hdr)
    print("-" * len(hdr))

    def loss_fn(m, x, y):
        return m.loss(x, y)
    lg = nn.value_and_grad(model, loss_fn)

    for B, T in grid:
        if T > cfg.seq_len:
            # build a temporary model at this seq_len for the score-matrix cost
            pass  # seq_len only affects mask size here; model handles any T
        x = mx.random.randint(0, cfg.vocab_size, (B, T))
        y = mx.random.randint(0, cfg.vocab_size, (B, T))
        mx.eval(x, y)

        if args.backward:
            fn = lambda: lg(model, x, y)
        else:
            fn = lambda: model(x)

        dt = time_fn(fn)

        # FLOPs: forward = 2*N*tokens + attention; backward ~ 2x forward
        tokens = B * T
        fwd_flops = 2 * n_params * tokens + cfg.n_layers * 4 * T * tokens
        flops = fwd_flops * (3 if args.backward else 1)

        acts = cfg.n_layers * B * T * cfg.d_model  # dominant activation term
        byts = bytes_moved(n_params, acts, args.backward)

        achieved_flops = flops / dt
        achieved_bw = byts / dt
        intensity = flops / byts
        mfu = achieved_flops / PEAK_FLOPS_FP16
        bound = "compute" if intensity > RIDGE_POINT else "memory"

        print(f"{B:>5} {T:>5} {dt*1000:>8.1f} {achieved_flops/1e12:>8.2f} "
              f"{achieved_bw/1e9:>7.0f} {intensity:>7.0f} {mfu*100:>5.1f}% {bound:>8}")

    print(f"\nridge point = {RIDGE_POINT:.0f} FLOP/byte "
          f"(intensity below this => memory-bound)")


if __name__ == "__main__":
    main()
