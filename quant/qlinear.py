"""Quantized Linear layers.

`QuantizedLinear` stores group-wise quantized weights (from quant.quantize) and
dequantizes them inside the forward matmul. This is the simplest correct
approach — "weight-only" quantization: weights live compressed in memory, get
expanded to fp just before the matmul. At batch=1 decode that still wins,
because the bottleneck is *reading the weights from memory*, and we read 4-8x
fewer bytes.

We also expose `native_quantized_linear`, a thin wrapper over MLX's fused
`nn.QuantizedLinear`, so bench.py can compare our explicit scheme against the
optimized kernel for both speed and quality.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .quantize import quantize, dequantize


class QuantizedLinear(nn.Module):
    """Weight-only group-wise quantized linear: y = x @ dequant(W).T (+ b)."""

    def __init__(self, weight: mx.array, bias: mx.array | None = None,
                 bits: int = 4, group_size: int = 64, symmetric: bool = False):
        super().__init__()
        codes, scale, zero, meta = quantize(weight, bits, group_size, symmetric)
        # stored compressed: codes are uint32 here (packing is a storage detail;
        # the memory-traffic win is modelled in roofline_quant.py)
        self.codes = codes
        self.scale = scale
        self.zero = zero
        if bias is not None:
            self.bias = bias
        self._bits = bits
        self._group_size = group_size
        self._shape = meta["shape"]

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4, group_size: int = 64,
                    symmetric: bool = False) -> "QuantizedLinear":
        bias = linear["bias"] if "bias" in linear else getattr(linear, "bias", None)
        return cls(linear.weight, bias, bits, group_size, symmetric)

    def _meta(self) -> dict:
        return {"bits": self._bits, "group_size": self._group_size,
                "shape": self._shape}

    def __call__(self, x: mx.array) -> mx.array:
        w = dequantize(self.codes, self.scale, self.zero, self._meta())
        y = x @ w.T
        if "bias" in self:
            y = y + self.bias
        return y


def native_quantized_linear(weight: mx.array, bias: mx.array | None,
                            bits: int = 4, group_size: int = 64) -> nn.Module:
    """MLX's fused QuantizedLinear, initialised from an existing weight.

    Used as the optimized-kernel baseline in benchmarks.
    """
    out_features, in_features = weight.shape
    ql = nn.QuantizedLinear(in_features, out_features, bias=bias is not None,
                            group_size=group_size, bits=bits)
    # overwrite the random init with the real (quantized) weight
    w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    ql.weight = w_q
    ql.scales = scales
    ql.biases = biases
    if bias is not None:
        ql.bias = bias
    return ql
