# MLX Quantization & Kernel Benchmarking

From-scratch weight quantization (INT8 / INT4, group-wise affine) for a
transformer in Apple MLX, with a rigorous **quality ↔ speed ↔ memory** benchmark
and a roofline analysis of why it works.

> Project #2 of [`scaling-from-scratch`](https://github.com/jadoon200/scaling-from-scratch).
> Project #1 (`mlx-tiny-transformer` branch) is the from-scratch transformer +
> scaling laws. This branch goes "below the stack" — the [frontier-lab path](https://vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html)
> calls out quantization and kernels as high-signal, low-compute work.

## The core idea

Autoregressive decode at batch=1 is **memory-bandwidth-bound**: every generated
token streams the entire weight matrix from memory. So decode speed is set by
`weight_bytes / peak_bandwidth`, not by FLOPs. Shrink the weights and you go
faster — almost linearly, in the memory-bound regime:

| precision | bytes/weight | effective BW vs fp16 | expected decode speedup |
|---|---|---|---|
| fp16 | 2 | 1× | 1× |
| INT8 | 1 | 2× | ~2× |
| INT4 | 0.5 | 4× | ~4× |

The price is quality. This project measures **both**, never one alone.

## Quantization scheme (group-wise affine)

Weights are quantized in groups of `G` along the input dimension; each group has
its own scale `s` and zero-point `z`:

```
q = clamp(round(w / s + z), 0, 2^bits - 1)      # integer code
w ≈ s * (q - z)                                  # dequantized
s = (w_max - w_min) / (2^bits - 1)               # per-group scale
```

Smaller `G` → better fidelity but more scale/zero overhead. The *effective*
bits/weight counts that overhead (`python config.py` prints the table).

## Status

| component | state |
|---|---|
| `quant/quantize.py` — group-wise INT8/INT4 quant, INT4 packing | ✅ built + verified |
| `config.py` — model + quant config, effective-bits accounting | ✅ |
| `quant/qlinear.py` — quantized Linear (dequant-matmul) + MLX-native compare | ⬜ next |
| `model.py` — compact transformer eval target | ⬜ |
| `bench.py` — perplexity / tokens-per-sec / memory vs bits | ⬜ |
| `roofline_quant.py` — quantization vs the memory roofline | ⬜ |
| `report.py` — figures + RESULTS.md | ⬜ |

### Verified so far
- Round-trip error: INT8 ≈ 0.5% rel-Frobenius, INT4 ≈ 8–10% (the expected cliff)
- Smaller group size lowers error (g=32: 0.0047 vs g=128: 0.0059 at INT8)
- INT4 pack/unpack is lossless; 8× compression vs fp32 weights

## Setup

Reuses project #1's conda env (identical deps):
```bash
conda activate mlx-transformer    # or: pip install -r requirements.txt
python config.py                  # effective-bits table
```

## Build plan
1. `qlinear.py`: a `QuantizedLinear` that stores packed codes + scales and
   dequantizes in the forward matmul; validate output parity against fp16 and
   against `mx.quantized_matmul`.
2. `model.py`: a small transformer; load/init weights, swap Linear → QuantizedLinear.
3. `bench.py`: sweep {fp16, INT8, INT4} × group sizes; record perplexity on
   FineWeb-Edu val, decode tokens/sec, and footprint.
4. `roofline_quant.py` + `report.py`: plot the speed/quality Pareto and the
   memory-roofline shift.
