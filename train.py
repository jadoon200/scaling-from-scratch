"""Training loop with loss / step-time / tokens-per-sec / MFU logging.

MFU (model FLOPs utilization) is the headline metric: achieved FLOP/s divided by
the chip's peak FLOP/s. It tells you how much of the hardware you're actually
using. Everything is logged every `log_every` steps.

Run:
    python train.py                       # uses ModelConfig() + TrainConfig() defaults
    python train.py --preset 1M --steps 2000
    python train.py --batch-size 32 --lr 6e-4
"""

from __future__ import annotations

import argparse
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from config import ModelConfig, TrainConfig, SCALING_PRESETS, PEAK_FLOPS_FP16
from data.dataset import TokenDataset, iterate_batches
from model import Transformer


def cosine_lr(step: int, tc: TrainConfig) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if step >= tc.max_steps:
        return tc.min_lr
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return tc.min_lr + coeff * (tc.lr - tc.min_lr)


def evaluate(model: Transformer, val_ds: TokenDataset, tc: TrainConfig) -> float:
    model.eval()
    batches = iterate_batches(val_ds, tc.batch_size, seed=tc.seed + 1)
    total = 0.0
    for _ in range(tc.eval_steps):
        x, y = next(batches)
        loss = model.loss(x, y)
        mx.eval(loss)
        total += loss.item()
    model.train()
    return total / tc.eval_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(SCALING_PRESETS), default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    cfg = SCALING_PRESETS[args.preset] if args.preset else ModelConfig()
    tc = TrainConfig()
    if args.steps is not None: tc.max_steps = args.steps
    if args.batch_size is not None: tc.batch_size = args.batch_size
    if args.lr is not None: tc.lr = args.lr
    if args.data_dir is not None: tc.data_dir = args.data_dir
    if args.run_name is not None: tc.run_name = args.run_name

    metrics = train_model(cfg, tc, verbose=True, save=True)
    print(f"\nfinal val_loss {metrics['val_loss']:.3f} "
          f"(ppl {math.exp(metrics['val_loss']):.1f}) | "
          f"avg MFU {metrics['avg_mfu']*100:.1f}%")


def train_model(cfg: ModelConfig, tc: TrainConfig,
                verbose: bool = True, save: bool = False) -> dict:
    """Train one model and return final metrics. Reused by scaling.py.

    Returns dict with: val_loss, train_loss, n_params, n_params_non_embedding,
    tokens_seen, avg_mfu.
    """
    mx.random.seed(tc.seed)

    train_ds = TokenDataset(os.path.join(tc.data_dir, "train.bin"), cfg.seq_len)
    val_ds = TokenDataset(os.path.join(tc.data_dir, "val.bin"), cfg.seq_len)
    batches = iterate_batches(train_ds, tc.batch_size, seed=tc.seed)

    model = Transformer(cfg)
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    tokens_per_step = tc.batch_size * cfg.seq_len * tc.grad_accum_steps

    if verbose:
        print(f"model: {n_params/1e6:.2f}M params "
              f"({cfg.n_params_non_embedding/1e6:.2f}M non-embed)  "
              f"d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads}")
        print(f"data:  {train_ds.n_tokens/1e6:.2f}M train tokens, seq_len={cfg.seq_len}")
        print(f"step:  {tokens_per_step:,} tokens/step  "
              f"({6*n_params*tokens_per_step/1e9:.1f} GFLOPs/step est.)")
        print(f"peak:  {PEAK_FLOPS_FP16/1e12:.1f} TFLOP/s fp16\n")

    opt = optim.AdamW(learning_rate=tc.lr, betas=[tc.beta1, tc.beta2],
                      weight_decay=tc.weight_decay)

    def loss_fn(model, x, y):
        return model.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y, lr):
        opt.learning_rate = lr
        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = optim.clip_grad_norm(grads, tc.grad_clip)
        opt.update(model, grads)
        return loss, gnorm

    if save:
        os.makedirs(tc.ckpt_dir, exist_ok=True)
    model.train()
    running_t = 0.0
    mfu_sum, mfu_n = 0.0, 0
    last_loss = float("nan")

    for it in range(tc.max_steps):
        lr = cosine_lr(it, tc)
        x, y = next(batches)

        mx.eval(x, y)  # ensure inputs ready before we start the clock
        t0 = time.perf_counter()
        loss, gnorm = step(x, y, lr)
        mx.eval(loss, model.parameters(), opt.state)  # force the step to complete
        dt = time.perf_counter() - t0
        running_t += dt

        if it % tc.log_every == 0:
            step_ms = running_t / max(1, (tc.log_every if it else 1)) * 1000
            running_t = 0.0
            toks_per_sec = tokens_per_step / (step_ms / 1000)
            achieved_flops = 6 * n_params * tokens_per_step / (step_ms / 1000)
            mfu = achieved_flops / PEAK_FLOPS_FP16
            last_loss = loss.item()
            if it > 0:  # skip step 0 (kernel compilation) in the MFU average
                mfu_sum += mfu; mfu_n += 1
            if verbose:
                print(f"step {it:>5} | loss {last_loss:6.3f} | lr {lr:.2e} | "
                      f"gnorm {gnorm.item():5.2f} | {step_ms:6.1f} ms | "
                      f"{toks_per_sec/1e3:6.1f}k tok/s | MFU {mfu*100:4.1f}%")

        if verbose and it > 0 and it % tc.eval_every == 0:
            vl = evaluate(model, val_ds, tc)
            print(f"  >> eval step {it}: val_loss {vl:.3f} (ppl {math.exp(vl):.1f})")

        if save and it > 0 and it % tc.ckpt_every == 0:
            path = os.path.join(tc.ckpt_dir, f"{tc.run_name}_step{it}.safetensors")
            model.save_weights(path)
            if verbose:
                print(f"  >> saved {path}")

    val_loss = evaluate(model, val_ds, tc)
    if save:
        path = os.path.join(tc.ckpt_dir, f"{tc.run_name}_final.safetensors")
        model.save_weights(path)
        if verbose:
            print(f"saved {path}")

    return {
        "val_loss": val_loss,
        "train_loss": last_loss,
        "n_params": n_params,
        "n_params_non_embedding": cfg.n_params_non_embedding,
        "tokens_seen": tc.max_steps * tokens_per_step,
        "avg_mfu": mfu_sum / max(1, mfu_n),
    }


if __name__ == "__main__":
    main()
