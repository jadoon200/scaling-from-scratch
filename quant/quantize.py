"""From-scratch group-wise affine weight quantization.

A weight matrix is quantized in groups of `group_size` along the input
dimension. Each group gets its own affine map:

    q = clamp(round(w / s + z), 0, 2^bits - 1)     # integer codes
    w ≈ s * (q - z)                                 # dequantized

with per-group scale `s = (w_max - w_min) / (2^bits - 1)` and zero-point
`z = round(-w_min / s)` (asymmetric), or `z = 2^(bits-1)` with a symmetric
range. INT4 codes are packed two-per-byte.

This mirrors what `mx.quantize` does with a fused kernel; we implement it
explicitly to understand it, then benchmark against the native path.
"""

from __future__ import annotations

import mlx.core as mx


def quantize(w: mx.array, bits: int = 4, group_size: int = 64,
             symmetric: bool = False):
    """Quantize a 2D weight (out, in) group-wise along the input dim.

    Returns (codes, scales, zeros, meta) where codes is uint32 holding the
    integer codes (unpacked, one per weight — packing is a storage concern
    handled by pack_int4), and scales/zeros are (out, in/group_size).
    """
    assert w.ndim == 2, "expects a 2D weight"
    out, in_dim = w.shape
    assert in_dim % group_size == 0, "in_dim must be divisible by group_size"
    qmax = (1 << bits) - 1

    g = w.reshape(out, in_dim // group_size, group_size)
    w_min = g.min(axis=-1, keepdims=True)
    w_max = g.max(axis=-1, keepdims=True)

    if symmetric:
        absmax = mx.maximum(mx.abs(w_min), mx.abs(w_max))
        scale = (2 * absmax) / qmax
        zero = mx.full(scale.shape, (qmax + 1) // 2, dtype=mx.float32)
    else:
        scale = (w_max - w_min) / qmax
        zero = mx.round(-w_min / mx.maximum(scale, 1e-12))

    scale = mx.maximum(scale, 1e-12)
    codes = mx.clip(mx.round(g / scale + zero), 0, qmax).astype(mx.uint32)
    codes = codes.reshape(out, in_dim)
    meta = {"bits": bits, "group_size": group_size, "shape": (out, in_dim),
            "symmetric": symmetric}
    return codes, scale, zero, meta


def dequantize(codes: mx.array, scale: mx.array, zero: mx.array, meta: dict):
    """Reconstruct the float weight from integer codes + per-group params."""
    out, in_dim = meta["shape"]
    g = meta["group_size"]
    c = codes.reshape(out, in_dim // g, g).astype(mx.float32)
    w = scale * (c - zero)
    return w.reshape(out, in_dim)


def quantization_error(w: mx.array, bits: int = 4, group_size: int = 64,
                       symmetric: bool = False) -> dict:
    """Round-trip and report error metrics."""
    codes, scale, zero, meta = quantize(w, bits, group_size, symmetric)
    w_hat = dequantize(codes, scale, zero, meta)
    err = w - w_hat
    rmse = mx.sqrt(mx.mean(err * err))
    max_abs = mx.max(mx.abs(err))
    # relative Frobenius error
    rel = mx.sqrt(mx.sum(err * err)) / mx.sqrt(mx.sum(w * w))
    mx.eval(rmse, max_abs, rel)
    return {"rmse": rmse.item(), "max_abs": max_abs.item(),
            "rel_fro": rel.item(),
            "max_code": int((1 << bits) - 1)}


# --------------------------------------------------------------- INT4 packing
def pack_int4(codes: mx.array) -> mx.array:
    """Pack INT4 codes (values 0..15) two-per-byte. codes: (out, in), in even."""
    out, in_dim = codes.shape
    assert in_dim % 2 == 0
    c = codes.astype(mx.uint32)
    lo = c[:, 0::2]
    hi = c[:, 1::2]
    return (lo | (hi << 4)).astype(mx.uint8)


def unpack_int4(packed: mx.array, in_dim: int) -> mx.array:
    """Inverse of pack_int4 -> (out, in_dim) uint32 codes."""
    out = packed.shape[0]
    p = packed.astype(mx.uint32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    codes = mx.zeros((out, in_dim), dtype=mx.uint32)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    return codes
