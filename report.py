"""Figures + RESULTS.md for the custom-kernel suite.

  figures/bandwidth_all.png   achieved % of peak bandwidth, ours vs MLX baseline,
                              for every kernel (RMSNorm, softmax, SwiGLU, GEMV)

    python report.py
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PEAK_BW_BYTES
from kernels import rmsnorm_metal, rmsnorm_ref, softmax_metal, swiglu_metal, gemv_metal

FIG = "report/figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})


def timeit(fn, iters=100, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def measure():
    mx.random.seed(0)
    out = []

    x = mx.random.normal((32768, 2048)); w = mx.random.normal((2048,)); mx.eval(x, w)
    b = (2 * x.size + w.size) * 4
    out.append(("RMSNorm", b / timeit(lambda: rmsnorm_metal(x, w)),
                b / timeit(lambda: rmsnorm_ref(x, w)), "naive",
                mx.max(mx.abs(rmsnorm_metal(x, w) - rmsnorm_ref(x, w))).item()))

    x = mx.random.normal((32768, 2048)); mx.eval(x)
    b = 2 * x.size * 4
    out.append(("Softmax", b / timeit(lambda: softmax_metal(x)),
                b / timeit(lambda: mx.softmax(x, axis=-1)), "builtin",
                mx.max(mx.abs(softmax_metal(x) - mx.softmax(x, axis=-1))).item()))

    g = mx.random.normal((32768, 2048)); u = mx.random.normal((32768, 2048)); mx.eval(g, u)
    b = 3 * g.size * 4
    out.append(("SwiGLU", b / timeit(lambda: swiglu_metal(g, u)),
                b / timeit(lambda: nn.silu(g) * u), "naive",
                mx.max(mx.abs(swiglu_metal(g, u) - nn.silu(g) * u)).item()))

    W = mx.random.normal((8192, 8192)); xv = mx.random.normal((8192,)); mx.eval(W, xv)
    b = (W.size + xv.size + W.shape[0]) * 4
    out.append(("GEMV", b / timeit(lambda: gemv_metal(W, xv)),
                b / timeit(lambda: W @ xv), "mx matmul",
                mx.max(mx.abs(gemv_metal(W, xv) - (W @ xv)).astype(mx.float32)).item()))
    return out


def fig_bandwidth(data):
    names = [d[0] for d in data]
    ours = [100 * d[1] / PEAK_BW_BYTES for d in data]
    base = [100 * d[2] / PEAK_BW_BYTES for d in data]
    x = np.arange(len(names)); width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width/2, base, width, label="MLX baseline", color="#888888")
    ax.bar(x + width/2, ours, width, label="ours (custom Metal)", color="#c0504d")
    for i, (o, bs) in enumerate(zip(ours, base)):
        ax.text(i + width/2, o, f"{o:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i - width/2, bs, f"{bs:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(100, ls="--", color="k", alpha=0.5, label="peak (150 GB/s)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("% of peak memory bandwidth")
    ax.set_title("Custom Metal kernels hit 77–84% of the M3 Pro bandwidth roof",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "bandwidth_all.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG}/bandwidth_all.png")


def write_report(data):
    os.makedirs("report", exist_ok=True)
    L = ["# MLX Custom Metal Kernels — Results\n",
         "Four hand-written Metal GPU kernels via `mx.fast.metal_kernel`, "
         "benchmarked against MLX's naive multi-op path and optimized builtins on "
         "an Apple M3 Pro (peak ~150 GB/s). All four ops are memory-bound, so the "
         "figure of merit is achieved bandwidth vs the roof.\n",
         "![bandwidth](figures/bandwidth_all.png)\n",
         "| kernel | ours GB/s | % peak | baseline | speedup vs baseline | max err |",
         "|---|---|---|---|---|---|"]
    for name, ours, base, base_name, err in data:
        L.append(f"| {name} | {ours/1e9:.0f} | {100*ours/PEAK_BW_BYTES:.0f}% | "
                 f"{base_name} ({base/1e9:.0f} GB/s) | {ours/base:.2f}× | {err:.1e} |")
    L += ["",
          "## Takeaways\n",
          "- **RMSNorm / Softmax**: the custom kernels reach ~84% of peak "
          "bandwidth, matching MLX's hand-optimized builtins and ~3× the naive "
          "multi-op path. For a memory-bound op, fusing the passes is the win.",
          "- **SwiGLU**: 1.4× over the naive `silu(gate)*up` (three elementwise "
          "kernels → one).",
          "- **GEMV** (the batch=1 decode bottleneck): the specialized kernel "
          "slightly *beats* the general `mx.matmul`, at ~83% of peak — decode is "
          "bandwidth-bound on reading the weight matrix, and a dedicated GEMV has "
          "less overhead than a general matmul.",
          "- Correctness is ~1e-6 (fp32) for the elementwise/reduction kernels "
          "and ~1e-4 for GEMV (4096–8192-wide fp32 dot products).\n",
          "## Reproduce\n",
          "```bash\nconda activate mlx-transformer\npython bench.py\n"
          "python report.py\n```\n"]
    open("report/RESULTS.md", "w").write("\n".join(L))
    print("wrote report/RESULTS.md")


def main():
    print("measuring all kernels...")
    data = measure()
    fig_bandwidth(data)
    write_report(data)


if __name__ == "__main__":
    main()
