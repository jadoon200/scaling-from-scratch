"""KV cache for autoregressive decoding.

During generation we process one token at a time. Without a cache, each new
token would re-attend over the whole prefix from scratch — O(T^2) recompute.
The cache stores past K and V so each step only projects the *new* token's K/V
and appends them: O(T) per step.

This is the single biggest inference speedup, and the KV cache is also the main
memory-bandwidth consumer at decode time — which is exactly why GQA (fewer KV
heads) matters for serving.

Interface expected by Attention:
    cache.offset                  -> current sequence length (for RoPE offset)
    cache.update_and_fetch(k, v)  -> appends, returns full (k, v) so far
"""

import mlx.core as mx


class KVCache:
    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        # k, v: (B, n_kv_heads, T_new, head_dim)
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values


def make_cache(n_layers: int) -> list[KVCache]:
    return [KVCache() for _ in range(n_layers)]
