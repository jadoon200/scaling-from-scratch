"""Kernel tests.

Two tiers:
  - portable: sanity-check the pure-MLX *reference* implementations on the CPU
    device (these define correctness; they run anywhere, incl. CI).
  - @gpu: each custom Metal kernel must match its reference. Metal-only, so these
    run locally on Apple Silicon and skip in CI.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from kernels import (rmsnorm_metal, rmsnorm_ref, softmax_metal, softmax_ref,
                     swiglu_metal, swiglu_ref, gemv_metal, gemv_ref,
                     attention_metal, attention_ref)
from kernels.qgemv import quantize_weight, dequantize_weight, qgemv_metal
from tests.conftest import requires_gpu

gpu = pytest.mark.gpu


# ---------------- portable: reference-implementation sanity (CPU) -------------

def test_rmsnorm_ref_gives_unit_rms():
    mx.set_default_device(mx.cpu); mx.random.seed(0)
    x = mx.random.normal((32, 256)) * 5.0
    w = mx.ones((256,))
    y = rmsnorm_ref(x, w, 1e-5)
    rms = mx.sqrt(mx.mean(y ** 2, axis=-1)); mx.eval(rms)
    assert mx.max(mx.abs(rms - 1.0)).item() < 1e-2


def test_softmax_ref_rows_sum_to_one():
    mx.set_default_device(mx.cpu); mx.random.seed(0)
    x = mx.random.normal((16, 512))
    s = softmax_ref(x).sum(-1); mx.eval(s)
    assert mx.max(mx.abs(s - 1.0)).item() < 1e-5


def test_swiglu_ref_matches_formula():
    mx.set_default_device(mx.cpu); mx.random.seed(0)
    g = mx.random.normal((8, 64)); u = mx.random.normal((8, 64))
    manual = (g * mx.sigmoid(g)) * u
    diff = mx.max(mx.abs(swiglu_ref(g, u) - manual)); mx.eval(diff)
    assert diff.item() < 1e-5


def test_gemv_ref_equals_matmul():
    mx.set_default_device(mx.cpu); mx.random.seed(0)
    W = mx.random.normal((128, 256)); x = mx.random.normal((256,))
    diff = mx.max(mx.abs(gemv_ref(W, x) - (W @ x))); mx.eval(diff)
    assert diff.item() < 1e-4


# ---------------- @gpu: custom kernel == reference ----------------------------

@requires_gpu
@gpu
def test_rmsnorm_kernel_matches_reference():
    mx.set_default_device(mx.gpu); mx.random.seed(0)
    x = mx.random.normal((128, 512)); w = mx.random.normal((512,))
    err = mx.max(mx.abs(rmsnorm_metal(x, w) - rmsnorm_ref(x, w))); mx.eval(err)
    assert err.item() < 1e-4


@requires_gpu
@gpu
def test_softmax_kernel_matches_reference():
    mx.set_default_device(mx.gpu); mx.random.seed(0)
    x = mx.random.normal((128, 1024))
    err = mx.max(mx.abs(softmax_metal(x) - softmax_ref(x))); mx.eval(err)
    assert err.item() < 1e-5


@requires_gpu
@gpu
def test_swiglu_kernel_matches_reference():
    mx.set_default_device(mx.gpu); mx.random.seed(0)
    g = mx.random.normal((256, 1408)); u = mx.random.normal((256, 1408))
    err = mx.max(mx.abs(swiglu_metal(g, u) - swiglu_ref(g, u))); mx.eval(err)
    assert err.item() < 1e-5


@requires_gpu
@gpu
def test_gemv_kernel_matches_reference():
    mx.set_default_device(mx.gpu); mx.random.seed(0)
    W = mx.random.normal((4096, 4096)); x = mx.random.normal((4096,))
    err = mx.max(mx.abs(gemv_metal(W, x) - gemv_ref(W, x))); mx.eval(err)
    assert err.item() < 1e-3            # wide fp32 dot-product accumulation


@requires_gpu
@gpu
def test_attention_kernel_matches_reference():
    mx.set_default_device(mx.gpu); mx.random.seed(0)
    R, T, D = 64, 1024, 64
    q = mx.random.normal((R, D)); K = mx.random.normal((R, T, D)); V = mx.random.normal((R, T, D))
    err = mx.max(mx.abs(attention_metal(q, K, V) - attention_ref(q, K, V))); mx.eval(err)
    assert err.item() < 1e-4


@requires_gpu
@gpu
def test_qgemv_kernel_matches_dequant_matmul():
    mx.set_default_device(mx.gpu); mx.random.seed(0)
    M, K, G = 2048, 2048, 64
    W = mx.random.normal((M, K)); x = mx.random.normal((K,))
    for bits in (8, 4):
        codes, scale, zero = quantize_weight(W, bits, G)
        ref = dequantize_weight(codes, scale, zero, bits, G, M, K) @ x
        ours = qgemv_metal(codes, scale, zero, x, bits, G, M, K)
        err = mx.max(mx.abs(ours - ref)); mx.eval(err)
        assert err.item() < 1e-2, f"INT{bits}"
