"""Minimal trainer to produce weights for the quantization perplexity benchmark.

Not the focus of this project (project #1 is the real training rig) — just
enough to get a model that has learned something, so INT8/INT4 perplexity
degradation is measurable rather than noise.

    python train.py --steps 400
"""

from __future__ import annotations

import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from config import ModelConfig
from model import Transformer


def batches(path, seq_len, batch, seed=0):
    toks = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(toks) - seq_len - 1, size=batch)
        x = np.stack([toks[i:i+seq_len] for i in ix]).astype(np.int64)
        y = np.stack([toks[i+1:i+1+seq_len] for i in ix]).astype(np.int64)
        yield mx.array(x), mx.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data", type=str, default="data/fineweb/train.bin")
    ap.add_argument("--out", type=str, default="model.safetensors")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    args = ap.parse_args()

    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, seq_len=256)
    model = Transformer(cfg); mx.eval(model.parameters())
    n = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"model {n/1e6:.1f}M params, training {args.steps} steps")

    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.1)
    lg = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))
    data = batches(args.data, cfg.seq_len, args.batch)
    warmup = max(1, args.steps // 20)

    model.train()
    for it in range(args.steps):
        lr = args.lr * min(1.0, (it + 1) / warmup)
        lr *= 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.steps)))
        opt.learning_rate = lr
        x, y = next(data)
        loss, grads = lg(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        if it % max(1, args.steps // 10) == 0:
            print(f"step {it:>4} | loss {loss.item():.3f}")

    model.save_weights(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
