"""Benchmark custom Metal kernels: correctness, speedup, achieved bandwidth.

Compares three RMSNorm implementations:
  - naive: pure-MLX multi-op (square, mean, rsqrt, mul, mul) — several kernels
  - ours:  the fused custom Metal kernel (kernels/rmsnorm.py)
  - fused: mx.fast.rms_norm, MLX's hand-optimized builtin (the gold standard)

RMSNorm is memory-bound, so the figure of merit is achieved bandwidth: bytes
moved / time, against the ~150 GB/s roof. A good fused kernel approaches it; the
naive version wastes bandwidth on extra passes.

    python bench.py
"""

from __future__ import annotations

import time

import mlx.core as mx

from config import PEAK_BW_BYTES
from kernels.rmsnorm import rmsnorm_metal, rmsnorm_ref


def timeit(fn, iters=100, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def bench_rmsnorm():
    eps = 1e-5
    shapes = [(8 * 512, 512), (8 * 512, 1024), (32 * 1024, 2048)]
    print(f"{'rows x D':>14} {'impl':>7} {'us':>9} {'GB/s':>7} {'%peak':>6} "
          f"{'speedup':>8} {'maxerr':>9}")
    print("-" * 70)
    for rows, D in shapes:
        mx.random.seed(0)
        x = mx.random.normal((rows, D))
        w = mx.random.normal((D,))
        mx.eval(x, w)

        ref = rmsnorm_ref(x, w, eps)
        ours = rmsnorm_metal(x, w, eps)
        fused = mx.fast.rms_norm(x, w, eps)
        mx.eval(ref, ours, fused)
        err_ours = mx.max(mx.abs(ours - ref)).item()
        err_fused = mx.max(mx.abs(fused - ref)).item()

        # bytes: read x + write out + read w (fp32)
        bytes_moved = (2 * rows * D + D) * 4

        t_naive = timeit(lambda: rmsnorm_ref(x, w, eps))
        t_ours = timeit(lambda: rmsnorm_metal(x, w, eps))
        t_fused = timeit(lambda: mx.fast.rms_norm(x, w, eps))

        def line(tag, t, err):
            bw = bytes_moved / t
            print(f"{f'{rows}x{D}':>14} {tag:>7} {t*1e6:>9.1f} {bw/1e9:>7.0f} "
                  f"{100*bw/PEAK_BW_BYTES:>5.0f}% {t_naive/t:>7.2f}x {err:>9.1e}")

        line("naive", t_naive, 0.0)
        line("ours", t_ours, err_ours)
        line("fused", t_fused, err_fused)
        print()


if __name__ == "__main__":
    bench_rmsnorm()
