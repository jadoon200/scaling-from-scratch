"""Benchmark all custom Metal kernels: correctness, speedup, achieved bandwidth.

For each kernel we compare the custom Metal version against MLX's path (a naive
multi-op reference and/or the optimized builtin) and report achieved memory
bandwidth as a fraction of the ~150 GB/s roof. These ops are memory-bound, so
bandwidth utilization is the figure of merit.

    python bench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn

from config import PEAK_BW_BYTES
from kernels import (rmsnorm_metal, rmsnorm_ref, softmax_metal,
                     swiglu_metal, gemv_metal)


def timeit(fn, iters=100, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def _bw(bytes_moved, t):
    return bytes_moved / t


def row(name, t, bytes_moved, baseline_t, err):
    bw = _bw(bytes_moved, t)
    print(f"{name:>22} {t*1e6:>9.1f} {bw/1e9:>7.0f} {100*bw/PEAK_BW_BYTES:>5.0f}% "
          f"{baseline_t/t:>7.2f}x {err:>9.1e}")


def main():
    print(f"{'kernel / impl':>22} {'us':>9} {'GB/s':>7} {'%pk':>6} {'speedup':>8} {'err':>9}")
    print("-" * 70)

    # ---- RMSNorm: naive multi-op vs ours vs builtin ----
    mx.random.seed(0)
    x = mx.random.normal((32768, 2048)); w = mx.random.normal((2048,)); mx.eval(x, w)
    b = (2 * x.size + w.size) * 4
    ref = rmsnorm_ref(x, w); ours = rmsnorm_metal(x, w); mx.eval(ref, ours)
    e = mx.max(mx.abs(ours - ref)).item()
    tn = timeit(lambda: rmsnorm_ref(x, w))
    row("rmsnorm naive", tn, b, tn, 0.0)
    row("rmsnorm ours", timeit(lambda: rmsnorm_metal(x, w)), b, tn, e)
    row("rmsnorm builtin", timeit(lambda: mx.fast.rms_norm(x, w, 1e-5)), b, tn,
        mx.max(mx.abs(mx.fast.rms_norm(x, w, 1e-5) - ref)).item())
    print()

    # ---- Softmax: ours vs builtin (mx.softmax is already fused) ----
    x = mx.random.normal((32768, 2048)); mx.eval(x)
    b = 2 * x.size * 4
    ref = mx.softmax(x, axis=-1); ours = softmax_metal(x); mx.eval(ref, ours)
    e = mx.max(mx.abs(ours - ref)).item()
    tb = timeit(lambda: mx.softmax(x, axis=-1))
    row("softmax builtin", tb, b, tb, 0.0)
    row("softmax ours", timeit(lambda: softmax_metal(x)), b, tb, e)
    print()

    # ---- SwiGLU: naive multi-op vs ours ----
    g = mx.random.normal((32768, 2048)); u = mx.random.normal((32768, 2048)); mx.eval(g, u)
    b = 3 * g.size * 4  # read gate + read up + write out
    ref = nn.silu(g) * u; ours = swiglu_metal(g, u); mx.eval(ref, ours)
    e = mx.max(mx.abs(ours - ref)).item()
    tn = timeit(lambda: nn.silu(g) * u)
    row("swiglu naive", tn, b, tn, 0.0)
    row("swiglu ours", timeit(lambda: swiglu_metal(g, u)), b, tn, e)
    print()

    # ---- GEMV: ours vs mx matmul (the decode bottleneck) ----
    W = mx.random.normal((8192, 8192)); xv = mx.random.normal((8192,)); mx.eval(W, xv)
    b = (W.size + xv.size + W.shape[0]) * 4  # W dominates
    ref = W @ xv; ours = gemv_metal(W, xv); mx.eval(ref, ours)
    e = mx.max(mx.abs(ours - ref)).item()
    tm = timeit(lambda: W @ xv)
    row("gemv mx-matmul", tm, b, tm, 0.0)
    row("gemv ours", timeit(lambda: gemv_metal(W, xv)), b, tm, e)


if __name__ == "__main__":
    main()
