"""Shared test config + the GPU gate.

The custom kernels use mx.fast.metal_kernel, which is Metal-only — those tests
are marked @gpu and run locally on Apple Silicon. CI (no Metal GPU) runs the
portable tier: sanity checks on the pure-MLX reference implementations.

Run:  pytest          (everything, on Apple Silicon)
      pytest -m "not gpu"   (portable tier — what CI runs)
"""

import mlx.core as mx
import pytest

requires_gpu = pytest.mark.skipif(
    not mx.metal.is_available(), reason="needs an Apple Metal GPU"
)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a Metal GPU")
