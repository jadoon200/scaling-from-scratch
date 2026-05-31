"""Fused SwiGLU activation as a custom Metal kernel.

    out = silu(gate) * up,   silu(z) = z * sigmoid(z)

In MLX this is sigmoid, multiply, multiply — three elementwise kernels, each
streaming the tensors through memory. Fused into one kernel it's a single read
of gate and up and a single write of out. Pure elementwise → one thread per
element, bandwidth-bound.
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256


def _kernel(N: int):
    src = f"""
        uint i = thread_position_in_grid.x;
        if (i < {N}u) {{
            float g = (float)gate[i];
            float s = g / (1.0f + metal::exp(-g));   // silu
            out[i] = (T)(s * (float)up[i]);
        }}
    """
    return mx.fast.metal_kernel(name=f"swiglu_{N}",
                                input_names=["gate", "up"],
                                output_names=["out"], source=src)


def swiglu_metal(gate: mx.array, up: mx.array) -> mx.array:
    N = gate.size
    g = gate.reshape(-1)
    u = up.reshape(-1)
    grid = ((N + _THREADS - 1) // _THREADS) * _THREADS
    (out,) = _kernel(N)(
        inputs=[g, u], template=[("T", gate.dtype)],
        grid=(grid, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[g.shape], output_dtypes=[gate.dtype],
    )
    return out.reshape(gate.shape)


def swiglu_ref(gate: mx.array, up: mx.array) -> mx.array:
    import mlx.nn as nn
    return nn.silu(gate) * up


if __name__ == "__main__":
    mx.random.seed(0)
    g = mx.random.normal((256, 1408)); u = mx.random.normal((256, 1408))
    err = mx.max(mx.abs(swiglu_metal(g, u) - swiglu_ref(g, u)))
    mx.eval(err)
    print(f"swiglu max abs error: {err.item():.2e}",
          "PASS" if err.item() < 1e-5 else "FAIL")
