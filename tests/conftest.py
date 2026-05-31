"""Shared test fixtures + the GPU gate.

Tests split into two tiers:
  - portable: pure-MLX math that runs on the CPU device, so it passes anywhere
    (including GitHub's macOS CI runners).
  - gpu: anything that needs Metal (custom kernels / fused ops). Marked @gpu and
    skipped automatically when no Metal GPU is present.

Run everything locally:   pytest
Run only portable (CI):   pytest -m "not gpu"
"""

import mlx.core as mx
import pytest

# skip GPU-only tests when there's no Metal device (e.g. CI)
requires_gpu = pytest.mark.skipif(
    not mx.metal.is_available(), reason="needs an Apple Metal GPU"
)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a Metal GPU")
