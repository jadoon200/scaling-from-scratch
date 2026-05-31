"""Matrix-vector product (GEMV) as a custom Metal kernel — the decode bottleneck.

    y = W @ x,   W: (M, K),  x: (K,),  y: (M,)

Autoregressive decode at batch=1 is exactly this: each token multiplies the
weight matrix by a single activation vector. It is **memory-bound** — every
element of W is read once and used in a single multiply-add, so arithmetic
intensity ≈ 1 and time ≈ bytes(W) / bandwidth.

One threadgroup per output row computes that row's dot product, reducing the
K partial products in threadgroup memory (fp32 accumulation).
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256


def _kernel(K: int):
    src = f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        const uint K = {K};
        const uint NT = {_THREADS};
        threadgroup float red[{_THREADS}];

        float local = 0.0f;
        for (uint i = tid; i < K; i += NT)
            local += (float)W[row*K + i] * (float)x[i];
        red[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = NT/2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] += red[tid+s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) out[row] = (T)red[0];
    """
    return mx.fast.metal_kernel(name=f"gemv_{K}", input_names=["W", "x"],
                                output_names=["out"], source=src)


def gemv_metal(W: mx.array, x: mx.array) -> mx.array:
    M, K = W.shape
    (out,) = _kernel(K)(
        inputs=[W, x.reshape(K)], template=[("T", W.dtype)],
        grid=(M * _THREADS, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[(M,)], output_dtypes=[W.dtype],
    )
    return out


def gemv_ref(W: mx.array, x: mx.array) -> mx.array:
    return W @ x.reshape(-1)


if __name__ == "__main__":
    mx.random.seed(0)
    W = mx.random.normal((4096, 4096)); x = mx.random.normal((4096,))
    err = mx.max(mx.abs(gemv_metal(W, x) - gemv_ref(W, x)))
    mx.eval(err)
    print(f"gemv max abs error: {err.item():.2e}",
          "PASS" if err.item() < 1e-3 else "FAIL")
