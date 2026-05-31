"""Autoregressive text generation with a KV cache.

Two phases:
  1. Prefill: run the whole prompt through the model once, populating the KV
     cache and producing logits for the next token.
  2. Decode: feed one token at a time; each step only projects the new token's
     K/V (the cache holds the rest), so it's O(1) per step instead of O(T).

Sampling strategies:
  - greedy:  argmax (deterministic)
  - top-p:   nucleus sampling with temperature (the default)

At batch=1 decoding is memory-bandwidth-bound: every step streams all the
weights from unified memory to produce a single token. tokens/sec here is a
bandwidth measurement, not a compute one.

Run:
    python generate.py --prompt "Once upon a time" --max-tokens 100
    python generate.py --strategy greedy --ckpt checkpoints/run_final.safetensors
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from config import ModelConfig, SCALING_PRESETS
from data.tokenizer import Tokenizer
from model import Transformer, make_cache


def sample(logits: mx.array, strategy: str, temp: float, top_p: float) -> mx.array:
    # logits: (vocab,)
    if strategy == "greedy" or temp == 0.0:
        return mx.argmax(logits)

    logits = logits * (1.0 / temp)
    probs = mx.softmax(logits, axis=-1)

    if strategy == "top-p":
        # sort descending, keep the smallest set whose cumulative prob >= top_p
        idx = mx.argsort(-probs)
        sorted_probs = probs[idx]
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        # mask out the tail beyond top_p (shift so we always keep at least one)
        keep = cumsum - sorted_probs < top_p
        sorted_probs = mx.where(keep, sorted_probs, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = mx.random.categorical(mx.log(sorted_probs + 1e-10))
        return idx[choice]

    return mx.random.categorical(mx.log(probs + 1e-10))


def generate(model: Transformer, tok: Tokenizer, prompt: str,
             max_tokens: int, strategy: str, temp: float, top_p: float):
    model.eval()
    ids = tok.encode(prompt)
    x = mx.array([ids])  # (1, T)

    cache = make_cache(len(model.blocks))

    # --- prefill ---
    t0 = time.perf_counter()
    logits = model(x, cache=cache)
    next_tok = sample(logits[0, -1], strategy, temp, top_p)
    mx.eval(next_tok)
    prefill_dt = time.perf_counter() - t0

    out = [next_tok.item()]
    print(prompt, end="", flush=True)
    print(tok.decode([out[-1]]), end="", flush=True)

    # --- decode ---
    t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        x = mx.array([[out[-1]]])  # (1, 1)
        logits = model(x, cache=cache)
        next_tok = sample(logits[0, -1], strategy, temp, top_p)
        mx.eval(next_tok)
        tid = next_tok.item()
        if tid == tok.eot:
            break
        out.append(tid)
        print(tok.decode([tid]), end="", flush=True)
    decode_dt = time.perf_counter() - t0
    print("\n")

    n_decoded = len(out)
    print(f"--- prefill {len(ids)} tok in {prefill_dt*1000:.1f} ms "
          f"({len(ids)/prefill_dt:.0f} tok/s)")
    if n_decoded > 1:
        print(f"--- decode  {n_decoded-1} tok in {decode_dt*1000:.1f} ms "
              f"({(n_decoded-1)/decode_dt:.1f} tok/s, bandwidth-bound)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--ckpt", type=str, default="checkpoints/run_final.safetensors")
    ap.add_argument("--preset", choices=list(SCALING_PRESETS), default=None)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--strategy", choices=["greedy", "top-p"], default="top-p")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    cfg = SCALING_PRESETS[args.preset] if args.preset else ModelConfig()
    tok = Tokenizer()
    model = Transformer(cfg)
    model.load_weights(args.ckpt)
    mx.eval(model.parameters())

    generate(model, tok, args.prompt, args.max_tokens,
             args.strategy, args.temp, args.top_p)


if __name__ == "__main__":
    main()
