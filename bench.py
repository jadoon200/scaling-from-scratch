"""Quantization benchmark: memory, decode latency, and (optionally) perplexity.

Three precisions are compared:
  - fp16: plain nn.Linear baseline
  - ours: from-scratch QuantizedLinear (dequant-then-matmul)
  - native: MLX's fused nn.QuantizedLinear (reads packed weights in the kernel)

Key lesson this surfaces: naive dequant-then-matmul shrinks *storage* but NOT
the matmul's memory traffic — it materializes the fp weight before multiplying,
so it isn't faster. The fused native kernel reads the packed weights directly,
so it actually wins the memory-bound decode. "Below the stack" in one table.

Storage footprint is real for both quantized paths.

Run:
    python bench.py                       # memory + latency (no training needed)
    python bench.py --ckpt model.safetensors --data data/fineweb/val.bin  # + perplexity
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from config import ModelConfig, QuantConfig
from model import Transformer, quantize_model
from quant.qlinear import native_quantized_linear


def model_bytes(model: nn.Module) -> int:
    total = 0
    for _, p in tree_flatten(model.parameters()):
        total += p.size * p.dtype.size
    return total


def time_decode(linear_fn, x: mx.array, iters: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(linear_fn(x))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(linear_fn(x))
    return (time.perf_counter() - t0) / iters * 1000  # ms


def bench_linear_kernels(d_in=4096, d_out=4096):
    """Single large matmul at batch=1 (the decode bottleneck) across kernels."""
    mx.random.seed(0)
    w = mx.random.normal((d_out, d_in)).astype(mx.float16)
    x = mx.random.normal((1, d_in)).astype(mx.float16)
    mx.eval(w, x)

    fp = nn.Linear(d_in, d_out, bias=False)
    fp.weight = w
    print(f"\n=== decode latency, single {d_out}x{d_in} matmul, batch=1 ===")
    print(f"{'precision':>16} {'ms':>8} {'speedup':>8}")
    base = time_decode(lambda z: fp(z), x)
    print(f"{'fp16':>16} {base:>8.3f} {1.0:>7.2f}x")
    from quant.qlinear import QuantizedLinear
    for bits in (8, 4):
        ours = QuantizedLinear(w, None, bits=bits, group_size=64)
        t = time_decode(lambda z: ours(z), x)
        print(f"{('ours INT'+str(bits)):>16} {t:>8.3f} {base/t:>7.2f}x")
        nat = native_quantized_linear(w, None, bits=bits, group_size=64)
        tn = time_decode(lambda z: nat(z), x)
        print(f"{('native INT'+str(bits)):>16} {tn:>8.3f} {base/tn:>7.2f}x")


def bench_model_memory(cfg: ModelConfig):
    print("\n=== model storage footprint vs precision ===")
    print(f"{'precision':>12} {'MB':>8} {'vs fp16':>8}")
    base = Transformer(cfg); mx.eval(base.parameters())
    fp_bytes = model_bytes(base)
    # report fp16-equivalent baseline (params stored fp32 by default; halve it)
    fp16_mb = fp_bytes / 1e6 / 2
    print(f"{'fp16':>12} {fp16_mb:>8.1f} {1.0:>7.2f}x")
    for bits in (8, 4):
        for g in (64,):
            m = Transformer(cfg); mx.eval(m.parameters())
            quantize_model(m, bits=bits, group_size=g)
            mx.eval(m.parameters())
            qc = QuantConfig(bits=bits, group_size=g)
            # storage = codes at `bits` + scales/zeros overhead, embedding stays fp16
            eff = qc.effective_bits()
            # estimate: non-embedding weights at eff bits, embedding at 16
            n_emb = cfg.vocab_size * cfg.d_model
            n_total = sum(p.size for _, p in tree_flatten(m.parameters()))
            # crude: count Linear params via difference is messy; report analytic
            mb = (n_emb * 16 + (n_total - n_emb) * eff) / 8 / 1e6
            print(f"{('INT'+str(bits)+' g'+str(g)):>12} {mb:>8.1f} {fp16_mb/mb:>7.2f}x")


def perplexity(model, val_path: str, cfg: ModelConfig, n_batches=20, batch=8):
    toks = np.memmap(val_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(n_batches):
        ix = rng.integers(0, len(toks) - cfg.seq_len - 1, size=batch)
        x = np.stack([toks[i:i+cfg.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([toks[i+1:i+1+cfg.seq_len] for i in ix]).astype(np.int64)
        loss = model.loss(mx.array(x), mx.array(y)); mx.eval(loss)
        total += loss.item()
    import math
    return math.exp(total / n_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data", type=str, default="data/fineweb/val.bin")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    args = ap.parse_args()

    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, seq_len=256)
    bench_model_memory(cfg)
    bench_linear_kernels()

    if args.ckpt:
        import os
        print("\n=== perplexity vs precision (trained weights) ===")
        for bits in (None, 8, 4):
            m = Transformer(cfg)
            m.load_weights(args.ckpt)
            mx.eval(m.parameters())
            if bits is not None:
                quantize_model(m, bits=bits, group_size=64)
                mx.eval(m.parameters())
            if os.path.exists(args.data):
                ppl = perplexity(m, args.data, cfg)
                tag = "fp16" if bits is None else f"INT{bits}"
                print(f"  {tag:>6}: ppl {ppl:.1f}")
    else:
        print("\n(no --ckpt: skipping perplexity — needs trained weights to be "
              "meaningful)")


if __name__ == "__main__":
    main()
