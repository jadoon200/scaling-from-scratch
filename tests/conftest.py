"""Shared test config + the GPU gate.

Project #2's quantization math all runs on the CPU device, so the whole suite is
portable (runs on CI). The @gpu marker / requires_gpu gate is kept for
consistency with the other projects in case a Metal-only check is added.

Run:  pytest          (everything)
      pytest -m "not gpu"   (portable tier — what CI runs)
"""

import mlx.core as mx
import pytest

requires_gpu = pytest.mark.skipif(
    not mx.metal.is_available(), reason="needs an Apple Metal GPU"
)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires a Metal GPU")
