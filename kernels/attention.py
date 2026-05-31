"""Fused single-query attention as a custom Metal kernel (the Flash insight).

For each (batch·head) row r:
    scores[j] = (q · K[j]) * scale          # j = 0..T-1
    p         = softmax(scores)
    out       = Σ_j p[j] · V[j]

The Flash Attention insight: never write the (T) score vector to global memory.
This kernel keeps scores in **threadgroup (on-chip) memory**, does the softmax
reduction there, and streams K and V once each. One threadgroup per row.

This is the decode-time attention (one query vs T cached keys) — the
memory-bound regime that dominates autoregressive generation. The naive path
materializes the scores in global memory across several kernels (matmul,
softmax, matmul); fusing avoids that traffic.
"""

from __future__ import annotations

import math

import mlx.core as mx

_THREADS = 256


def _kernel(T: int, D: int, scale: float):
    src = f"""
        uint r   = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        const uint T = {T};
        const uint D = {D};
        const uint NT = {_THREADS};
        const float scale = {scale}f;

        threadgroup float scores[{T}];
        threadgroup float red[{_THREADS}];

        // scores[j] = (q . K[j]) * scale, track local max
        float lmax = -INFINITY;
        for (uint j = tid; j < T; j += NT) {{
            float s = 0.0f;
            for (uint e = 0; e < D; e++)
                s += (float)q[r*D + e] * (float)K[(r*T + j)*D + e];
            s *= scale;
            scores[j] = s;
            lmax = metal::max(lmax, s);
        }}
        red[tid] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = NT/2; k > 0; k >>= 1) {{
            if (tid < k) red[tid] = metal::max(red[tid], red[tid+k]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float m = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // exp(scores - m) in place, track local sum
        float lsum = 0.0f;
        for (uint j = tid; j < T; j += NT) {{
            float p = metal::exp(scores[j] - m);
            scores[j] = p;
            lsum += p;
        }}
        red[tid] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = NT/2; k > 0; k >>= 1) {{
            if (tid < k) red[tid] += red[tid+k];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float l = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // out[e] = (Σ_j p[j] * V[j,e]) / l   — each thread owns output dims e
        for (uint e = tid; e < D; e += NT) {{
            float acc = 0.0f;
            for (uint j = 0; j < T; j++)
                acc += scores[j] * (float)V[(r*T + j)*D + e];
            out[r*D + e] = (DT)(acc / l);
        }}
    """
    return mx.fast.metal_kernel(name=f"attn_{T}_{D}",
                                input_names=["q", "K", "V"],
                                output_names=["out"], source=src)


def attention_metal(q: mx.array, K: mx.array, V: mx.array, scale=None) -> mx.array:
    """q: (R, D), K/V: (R, T, D) -> out: (R, D)."""
    R, D = q.shape
    T = K.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    (out,) = _kernel(T, D, scale)(
        inputs=[q, K, V], template=[("DT", q.dtype)],
        grid=(R * _THREADS, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[(R, D)], output_dtypes=[q.dtype],
    )
    return out


def attention_ref(q, K, V, scale=None):
    R, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scores = (q[:, None, :] * K).sum(-1) * scale       # (R, T)
    p = mx.softmax(scores, axis=-1)
    return (p[:, :, None] * V).sum(1)                  # (R, D)


if __name__ == "__main__":
    mx.random.seed(0)
    R, T, D = 64, 1024, 64
    q = mx.random.normal((R, D)); K = mx.random.normal((R, T, D)); V = mx.random.normal((R, T, D))
    err = mx.max(mx.abs(attention_metal(q, K, V) - attention_ref(q, K, V)))
    mx.eval(err)
    print(f"attention max abs error: {err.item():.2e}",
          "PASS" if err.item() < 1e-4 else "FAIL")
