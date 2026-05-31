"""Download + tokenize a slice of FineWeb-Edu into a flat token array on disk.

FineWeb-Edu is a high-quality filtered web corpus. We stream a slice, tokenize
to GPT-2 BPE, concatenate with <|endoftext|> separators, and write a single
uint16 .bin file (50304 < 2^16, so uint16 holds every token id in 2 bytes).

The .bin is memory-mapped at train time, so we never load the whole corpus into
RAM — important on an 18 GB machine. A 90/10 train/val split is written as two
files.

Usage:
    python -m data.download --tokens 100_000_000
    python -m data.download --tokens 10_000_000 --name sample-10BT
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset

from .tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=100_000_000,
                    help="approximate total tokens to collect")
    ap.add_argument("--name", type=str, default="sample-10BT",
                    help="FineWeb-Edu config name (a sample subset)")
    ap.add_argument("--out", type=str, default="data/fineweb")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.name,
                      split="train", streaming=True)

    # Pre-allocate; trim at the end. uint16 because max id (50256) < 65536.
    buf = np.empty(args.tokens + 1_000_000, dtype=np.uint16)
    n = 0
    for i, doc in enumerate(ds):
        ids = tok.encode(doc["text"], add_eot=True)
        if n + len(ids) > len(buf):
            break
        buf[n:n + len(ids)] = ids
        n += len(ids)
        if i % 1000 == 0:
            print(f"\r{i:>7} docs  {n/1e6:7.2f}M / {args.tokens/1e6:.0f}M tokens",
                  end="", flush=True)
        if n >= args.tokens:
            break
    print()

    buf = buf[:n]
    n_val = int(n * args.val_frac)
    val, train = buf[:n_val], buf[n_val:]

    train.tofile(os.path.join(args.out, "train.bin"))
    val.tofile(os.path.join(args.out, "val.bin"))
    print(f"wrote {len(train)/1e6:.2f}M train + {len(val)/1e6:.2f}M val tokens "
          f"to {args.out}/  ({n*2/1e6:.1f} MB total)")


if __name__ == "__main__":
    main()
