"""Quantization correctness tests — all portable (run on the CPU device)."""

import mlx.core as mx
import mlx.nn as nn

from config import QuantConfig
from quant.quantize import quantization_error, quantize, pack_int4, unpack_int4
from quant.qlinear import QuantizedLinear, native_quantized_linear


def _w():
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    return mx.random.normal((512, 1024))


def test_int8_roundtrip_error_small():
    e = quantization_error(_w(), bits=8, group_size=64)
    assert e["rel_fro"] < 0.02            # INT8 ≈ 0.5%


def test_int4_roundtrip_error_bounded():
    e = quantization_error(_w(), bits=4, group_size=64)
    assert 0.02 < e["rel_fro"] < 0.15     # INT4 ≈ 8–10%, the expected cliff


def test_int8_more_accurate_than_int4():
    w = _w()
    e8 = quantization_error(w, bits=8, group_size=64)["rel_fro"]
    e4 = quantization_error(w, bits=4, group_size=64)["rel_fro"]
    assert e8 < e4


def test_smaller_group_lowers_error():
    w = _w()
    fine = quantization_error(w, bits=8, group_size=32)["rel_fro"]
    coarse = quantization_error(w, bits=8, group_size=128)["rel_fro"]
    assert fine < coarse                  # finer groups track the weights better


def test_int4_pack_unpack_lossless():
    w = _w()
    codes, scale, zero, meta = quantize(w, bits=4, group_size=64)
    packed = pack_int4(codes)
    restored = unpack_int4(packed, meta["shape"][1])
    mx.eval(packed, restored)
    assert bool(mx.all(codes == restored).item())
    assert packed.nbytes * 2 == codes.size   # two 4-bit codes per byte


def test_effective_bits_accounting():
    # INT4 with group 64 ⇒ 4 + (16+16)/64 = 4.5 effective bits/weight
    assert abs(QuantConfig(bits=4, group_size=64).effective_bits() - 4.5) < 1e-9
    assert abs(QuantConfig(bits=8, group_size=64).effective_bits() - 8.5) < 1e-9


def test_qlinear_matches_fp16_within_quant_error():
    mx.set_default_device(mx.cpu)
    lin = nn.Linear(1024, 512, bias=False); mx.eval(lin.parameters())
    x = mx.random.normal((4, 1024))
    ref = lin(x)
    q = QuantizedLinear.from_linear(lin, bits=8, group_size=64)
    y = q(x); mx.eval(ref, y)
    rel = (mx.sqrt(mx.sum((y - ref) ** 2)) / mx.sqrt(mx.sum(ref ** 2))).item()
    assert rel < 0.02                     # INT8 output error stays small


def test_from_scratch_qlinear_matches_native_kernel():
    """Our explicit QuantizedLinear should match MLX's fused kernel closely."""
    mx.set_default_device(mx.cpu)
    lin = nn.Linear(1024, 512, bias=False); mx.eval(lin.parameters())
    x = mx.random.normal((4, 1024))
    ours = QuantizedLinear.from_linear(lin, bits=8, group_size=64)(x)
    nat = native_quantized_linear(lin.weight, None, bits=8, group_size=64)(x)
    mx.eval(ours, nat)
    rel = (mx.sqrt(mx.sum((ours - nat) ** 2)) / mx.sqrt(mx.sum(nat ** 2))).item()
    assert rel < 0.02
