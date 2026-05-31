"""Transformer correctness tests.

Portable tests force the CPU device so they run on CI; the cached-decode test is
marked @gpu (it leans on the fused attention/rope path we exercise on Metal).
"""

import math

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from config import ModelConfig
from model import Transformer, make_cache
from tests.conftest import requires_gpu

gpu = pytest.mark.gpu


def _small():
    return ModelConfig(d_model=128, n_layers=2, n_heads=4, seq_len=64)


def test_analytic_param_count_matches_model():
    """cfg.n_params should match the real model to within the norm weights."""
    mx.set_default_device(mx.cpu)
    cfg = _small()
    m = Transformer(cfg); mx.eval(m.parameters())
    actual = sum(p.size for _, p in tree_flatten(m.parameters()))
    # difference is exactly the RMSNorm weights (2 per block + final), each d_model
    norm_params = (2 * cfg.n_layers + 1) * cfg.d_model
    assert actual == cfg.n_params + norm_params


def test_init_loss_near_ln_vocab():
    """A freshly-initialised LM should sit near uniform: loss ≈ ln(vocab)."""
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    cfg = _small()
    m = Transformer(cfg); mx.eval(m.parameters())
    x = mx.random.randint(0, cfg.vocab_size, (4, cfg.seq_len))
    loss = m.loss(x, x); mx.eval(loss)
    # untrained ⇒ roughly uniform; allow generous slack for a tiny model's
    # init variance, but catch a broken init (loss ~0 or ≫ ln(vocab))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 4.0


def test_forward_shape():
    mx.set_default_device(mx.cpu)
    cfg = _small()
    m = Transformer(cfg); mx.eval(m.parameters())
    x = mx.random.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    y = m(x); mx.eval(y)
    assert y.shape == (2, cfg.seq_len, cfg.vocab_size)


def test_non_embedding_params_scale_quadratically_with_d():
    """Doubling d_model ~4x the non-embedding params (≈12 d²; not exact because
    the SwiGLU d_ff rounds to a multiple of 64)."""
    a = ModelConfig(d_model=128, n_layers=4, n_heads=4)
    b = ModelConfig(d_model=256, n_layers=4, n_heads=4)
    ratio = b.n_params_non_embedding / a.n_params_non_embedding
    assert 3.5 < ratio < 4.3


@requires_gpu
@gpu
def test_cached_decode_matches_full_forward():
    """Incremental KV-cache decode must equal a full forward at the last pos."""
    cfg = _small()
    m = Transformer(cfg); mx.eval(m.parameters()); m.eval()
    ids = mx.array([[464, 47385, 10959, 16252, 351, 24061]])
    full = m(ids); mx.eval(full)
    cache = make_cache(cfg.n_layers)
    _ = m(ids[:, :5], cache=cache)
    step = m(ids[:, 5:6], cache=cache); mx.eval(step)
    diff = mx.abs(full[0, 5] - step[0, 0]).max().item()
    assert diff < 1e-3
