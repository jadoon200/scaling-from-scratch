"""Fused numerically-stable row softmax as a custom Metal kernel.

softmax(x)[i] = exp(x[i] - max) / sum_j exp(x[j] - max)

Naively this is max (reduction), subtract, exp, sum (reduction), divide — several
memory passes. Fused: one threadgroup per row does both reductions in shared
memory with a single read of x and single write of y. Memory-bound, so the win
is the avoided passes.
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256


def _kernel(D: int):
    src = f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        const uint D = {D};
        const uint NT = {_THREADS};
        threadgroup float red[{_THREADS}];

        // row max
        float m = -INFINITY;
        for (uint i = tid; i < D; i += NT) m = metal::max(m, (float)x[row*D+i]);
        red[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = NT/2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] = metal::max(red[tid], red[tid+s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rowmax = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // sum of exp(x - max)
        float sm = 0.0f;
        for (uint i = tid; i < D; i += NT) sm += metal::exp((float)x[row*D+i] - rowmax);
        red[tid] = sm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = NT/2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] += red[tid+s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rowsum = red[0];

        for (uint i = tid; i < D; i += NT)
            out[row*D+i] = (T)(metal::exp((float)x[row*D+i] - rowmax) / rowsum);
    """
    return mx.fast.metal_kernel(name=f"softmax_{D}", input_names=["x"],
                                output_names=["out"], source=src)


def softmax_metal(x: mx.array) -> mx.array:
    *lead, D = x.shape
    n = 1
    for s in lead:
        n *= s
    xr = x.reshape(n, D)
    (out,) = _kernel(D)(
        inputs=[xr], template=[("T", x.dtype)],
        grid=(n * _THREADS, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[xr.shape], output_dtypes=[x.dtype],
    )
    return out.reshape(*lead, D)


def softmax_ref(x: mx.array) -> mx.array:
    return mx.softmax(x, axis=-1)


if __name__ == "__main__":
    mx.random.seed(0)
    x = mx.random.normal((128, 1024))
    err = mx.max(mx.abs(softmax_metal(x) - softmax_ref(x)))
    mx.eval(err)
    print(f"softmax max abs error: {err.item():.2e}",
          "PASS" if err.item() < 1e-5 else "FAIL")
