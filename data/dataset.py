"""Memory-mapped token dataset + batching.

The .bin file (uint16 token stream from download.py) is memory-mapped, so we
pay zero RAM for the corpus and let the OS page in only the windows we touch.

A batch is built by sampling B random start positions and slicing T+1 contiguous
tokens at each: x = tokens[i:i+T], y = tokens[i+1:i+T+1] (next-token targets).
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np


class TokenDataset:
    def __init__(self, bin_path: str, seq_len: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"{bin_path} not found — run `python -m data.download` first."
            )
        # mmap so the corpus never fully enters RAM
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_tokens = len(self.tokens)
        if self.n_tokens <= seq_len + 1:
            raise ValueError(f"corpus ({self.n_tokens} tok) shorter than seq_len+1")

    def __len__(self) -> int:
        return self.n_tokens

    def batch(self, batch_size: int, rng: np.random.Generator) -> tuple[mx.array, mx.array]:
        # random start offsets, leaving room for T+1 tokens
        ix = rng.integers(0, self.n_tokens - self.seq_len - 1, size=batch_size)
        # gather windows (int64 for MLX embedding indexing)
        x = np.stack([self.tokens[i:i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.tokens[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        return mx.array(x), mx.array(y)


def iterate_batches(dataset: TokenDataset, batch_size: int, seed: int = 0):
    """Infinite generator of (x, y) batches."""
    rng = np.random.default_rng(seed)
    while True:
        yield dataset.batch(batch_size, rng)
