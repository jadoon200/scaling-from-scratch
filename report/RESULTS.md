# MLX Custom Metal Kernels — Results

Four hand-written Metal GPU kernels via `mx.fast.metal_kernel`, benchmarked against MLX's naive multi-op path and optimized builtins on an Apple M3 Pro (peak ~150 GB/s). All four ops are memory-bound, so the figure of merit is achieved bandwidth vs the roof.

![bandwidth](figures/bandwidth_all.png)

| kernel | ours GB/s | % peak | baseline | speedup vs baseline | max err |
|---|---|---|---|---|---|
| RMSNorm | 127 | 84% | naive (38 GB/s) | 3.30× | 1.9e-06 |
| Softmax | 126 | 84% | builtin (128 GB/s) | 0.98× | 7.5e-09 |
| SwiGLU | 115 | 76% | naive (81 GB/s) | 1.42× | 1.9e-06 |
| GEMV | 125 | 83% | mx matmul (122 GB/s) | 1.02× | 1.2e-04 |

## Takeaways

- **RMSNorm / Softmax**: the custom kernels reach ~84% of peak bandwidth, matching MLX's hand-optimized builtins and ~3× the naive multi-op path. For a memory-bound op, fusing the passes is the win.
- **SwiGLU**: 1.4× over the naive `silu(gate)*up` (three elementwise kernels → one).
- **GEMV** (the batch=1 decode bottleneck): the specialized kernel slightly *beats* the general `mx.matmul`, at ~83% of peak — decode is bandwidth-bound on reading the weight matrix, and a dedicated GEMV has less overhead than a general matmul.
- Correctness is ~1e-6 (fp32) for the elementwise/reduction kernels and ~1e-4 for GEMV (4096–8192-wide fp32 dot products).

## Reproduce

```bash
conda activate mlx-transformer
python bench.py
python report.py
```
