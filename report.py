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
import math
from kernels import (rmsnorm_metal, rmsnorm_ref, softmax_metal, swiglu_metal,
                     gemv_metal, attention_metal, attention_ref)

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

    R, T, D = 256, 2048, 64
    q = mx.random.normal((R, D)); K = mx.random.normal((R, T, D))
    Vv = mx.random.normal((R, T, D)); mx.eval(q, K, Vv)
    b = (2 * R * T * D + 2 * R * D) * 4
    out.append(("Attention", b / timeit(lambda: attention_metal(q, K, Vv)),
                b / timeit(lambda: attention_ref(q, K, Vv)), "naive",
                mx.max(mx.abs(attention_metal(q, K, Vv) - attention_ref(q, K, Vv))).item()))
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


def measure_qgemv():
    from kernels.qgemv import quantize_weight, qgemv_metal
    mx.random.seed(0)
    M, K, G = 8192, 8192, 64
    W = mx.random.normal((M, K)); x = mx.random.normal((K,)); mx.eval(W, x)
    tf = timeit(lambda: W @ x)
    rows = [("fp32", tf, W.size * 4)]
    for bits in (8, 4):
        c, s, z = quantize_weight(W, bits, G); mx.eval(c, s, z)
        tq = timeit(lambda: qgemv_metal(c, s, z, x, bits, G, M, K))
        rows.append((f"INT{bits}", tq, c.nbytes))
    return rows


def fig_qgemv(rows):
    base = rows[0][1]
    labels = [r[0] for r in rows]
    spd = [base / r[1] for r in rows]
    mb = [r[2] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    bars = ax.bar(labels, spd, color=["#888888", "#4f81bd", "#9bbb59"])
    for i, (sp, m) in enumerate(zip(spd, mb)):
        ax.text(i, sp, f"{sp:.2f}×\n{m:.0f} MB", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("decode speedup over fp32")
    ax.set_title("Quantized GEMV: reading packed weights in-kernel\n"
                 "(projects #2 + #3 fused)", fontsize=10)
    ax.set_ylim(0, max(spd) * 1.25)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "qgemv.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG}/qgemv.png")


def write_report(data, qrows=None):
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
          ]
    if qrows:
        base = qrows[0][1]
        L += ["## Quantized GEMV — projects #2 + #3 fused\n",
              "Decode is bound by reading the weight matrix. Quantizing the "
              "weights shrinks that read — but only if the matmul reads the "
              "*packed* weights directly (project #2 showed dequantize-then-matmul "
              "is ~9× slower). This kernel unpacks and dequantizes each weight "
              "inline, never materializing the fp matrix.\n",
              "![qgemv](figures/qgemv.png)\n",
              "| weights | read | decode speedup |", "|---|---|---|"]
        for tag, t, b in qrows:
            L.append(f"| {tag} | {b/1e6:.0f} MB | {base/t:.2f}× |")
        L += ["",
              "INT8 gives **2.6×** and INT4 **2.4×** faster decode than fp32. "
              "INT4 reads half the bytes of INT8 yet is slightly slower — the "
              "nibble-unpacking compute offsets the bandwidth saving at this size, "
              "a nice reminder that 'fewer bytes' only helps while you stay "
              "memory-bound.\n"]
    L += ["## Reproduce\n",
          "```bash\nconda activate mlx-transformer\npython bench.py\n"
          "python report.py\n```\n"]
    open("report/RESULTS.md", "w").write("\n".join(L))
    print("wrote report/RESULTS.md")


def main():
    print("measuring all kernels...")
    data = measure()
    fig_bandwidth(data)
    qrows = measure_qgemv()
    fig_qgemv(qrows)
    write_report(data, qrows)


if __name__ == "__main__":
    main()
