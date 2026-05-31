# MLX Custom Metal Kernels

Hand-written **Metal GPU kernels** for transformer ops via `mx.fast.metal_kernel`,
benchmarked against MLX's naive multi-op path and its hand-optimized builtins on
the Apple Silicon roofline.

> Project #3 of [`scaling-from-scratch`](https://github.com/jadoon200/scaling-from-scratch).
> #1 = transformer + scaling laws, #2 = quantization, #3 = the GPU code itself —
> the deepest "below the stack" rung of the [frontier-lab path](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html)
> (the Flash Attention insight is a kernel-fusion insight).

## The idea

Ops like RMSNorm, softmax, and activations are **memory-bound**. In a naive
framework each runs as several separate GPU kernels — every pass re-reads and
re-writes the activations, plus per-launch overhead. **Fusing** them into one
custom kernel cuts the memory traffic and the launches, which is the entire win
for a memory-bound op. The figure of merit isn't FLOPs — it's *achieved
bandwidth as a fraction of the ~150 GB/s roof*.

## Result: fused RMSNorm

```
y[i] = x[i] / sqrt(mean(x²) + eps) * w[i]
```

One threadgroup per row: threads cooperatively reduce the sum of squares in
threadgroup memory (fp32 accumulation), then write the normalized output — one
read of x, one write of y.

| size | naive | ours (custom) | builtin | ours speedup |
|---|---|---|---|---|
| 4096×512 | 16 GB/s | 34 GB/s | 51 GB/s | 2.1× |
| 4096×1024 | 30 GB/s | 78 GB/s | 81 GB/s | 2.6× |
| 32768×2048 | 38 GB/s | **127 GB/s (84% peak)** | 127 GB/s | **3.3×** |

![bandwidth](report/figures/bandwidth.png)

At scale the from-scratch Metal kernel reaches **84% of peak memory bandwidth**,
**matching MLX's hand-optimized builtin** and ~3.3× the naive version — correct
to ~1e-6 vs the reference. At small sizes the builtin's launch tuning still wins;
the gap closes as the problem grows.

## Layout

```
kernels/
  rmsnorm.py     fused RMSNorm: custom Metal kernel + pure-MLX reference
config.py        chip peak specs
bench.py         correctness + speedup + achieved bandwidth
report.py        figures + RESULTS.md
```

## Setup & usage

Reuses project #1's conda env:
```bash
conda activate mlx-transformer    # or: pip install -r requirements.txt
python kernels/rmsnorm.py         # correctness check
python bench.py                   # bandwidth + speedup table
python report.py                  # figures + RESULTS.md
```

See [`report/RESULTS.md`](report/RESULTS.md) for the full writeup.

## Next kernels
- Fused softmax (row-wise, online/streaming max+sum)
- Fused SwiGLU activation (`silu(gate) * up` in one pass)
- A batch=1 GEMV tuned for the decode bottleneck
