"""Fused RMSNorm as a custom Metal kernel.

RMSNorm:  y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]

Naively (pure MLX) this is several ops — square, mean (reduction), rsqrt,
multiply, multiply — each a separate GPU kernel that re-reads/writes the
activations. RMSNorm is memory-bound, so those extra passes and launches are
pure overhead. Fusing the whole thing into one kernel — one read of x, one
write of y, the reduction in threadgroup memory — is the win.

Design: one threadgroup per row. Threads cooperatively accumulate the sum of
squares over the row (strided), tree-reduce it in shared memory, then each
thread writes its normalized, weighted outputs. Reduction accumulates in fp32
even for fp16 I/O.
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256  # threads per row; must be a power of two for the tree reduction


def _kernel(D: int, eps: float):
    # D and eps are baked into the source so the kernel is self-contained
    # (no reliance on shape/stride conventions). MLX caches by source string.
    src = f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        const uint D = {D};
        const uint NT = {_THREADS};

        threadgroup float partial[{_THREADS}];

        float local = 0.0f;
        for (uint i = tid; i < D; i += NT) {{
            float v = (float)x[row * D + i];
            local += v * v;
        }}
        partial[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = NT / 2; s > 0; s >>= 1) {{
            if (tid < s) partial[tid] += partial[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv = metal::rsqrt(partial[0] / (float)D + {eps}f);
        for (uint i = tid; i < D; i += NT) {{
            out[row * D + i] = (T)((float)x[row * D + i] * inv * (float)w[i]);
        }}
    """
    return mx.fast.metal_kernel(
        name=f"rmsnorm_{D}",
        input_names=["x", "w"],
        output_names=["out"],
        source=src,
    )


def rmsnorm_metal(x: mx.array, w: mx.array, eps: float = 1e-5) -> mx.array:
    """Custom-kernel RMSNorm over the last dim. x: (..., D), w: (D,)."""
    *lead, D = x.shape
    n_rows = 1
    for s in lead:
        n_rows *= s
    xr = x.reshape(n_rows, D)
    kernel = _kernel(D, eps)
    (out,) = kernel(
        inputs=[xr, w],
        template=[("T", x.dtype)],
        grid=(n_rows * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[xr.shape],
        output_dtypes=[x.dtype],
    )
    return out.reshape(*lead, D)


def rmsnorm_ref(x: mx.array, w: mx.array, eps: float = 1e-5) -> mx.array:
    """Pure-MLX reference (the naive multi-op version)."""
    var = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    return (x.astype(mx.float32) * mx.rsqrt(var + eps)).astype(x.dtype) * w


if __name__ == "__main__":
    mx.random.seed(0)
    x = mx.random.normal((128, 512))
    w = mx.random.normal((512,))
    a = rmsnorm_metal(x, w); b = rmsnorm_ref(x, w)
    mx.eval(a, b)
    err = mx.max(mx.abs(a - b)).item()
    print(f"max abs error vs reference: {err:.2e}")
    print("PASS" if err < 1e-4 else "FAIL")
