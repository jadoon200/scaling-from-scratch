"""Quantized GEMV — the fusion of project #2 (quantization) and #3 (kernels).

Decode (batch=1) is bound by *reading the weight matrix*. Quantizing the weights
shrinks that read — INT8 is 1 byte/weight (4× less than fp32), INT4 is 0.5 byte
(8× less). But the win only materializes if the matmul reads the *packed* weights
directly: project #2 showed that dequantize-then-matmul is ~9× slower because it
expands the weight back to float first. This kernel does it right — it unpacks
and dequantizes each weight inline, in registers, and never materializes the
fp weight matrix.

    y = dequant(W_q) @ x,   group-wise affine quant:  w ≈ scale_g * (code - zero_g)
"""

from __future__ import annotations

import mlx.core as mx

_THREADS = 256


def quantize_weight(W: mx.array, bits: int, group_size: int):
    """Group-wise affine quantize a (M, K) weight. Returns (codes, scales, zeros).

    For INT4, codes are packed two-per-byte (uint8, shape (M, K/2)); for INT8
    codes are uint8 (M, K). scales/zeros are fp32 (M, K/group_size).
    """
    M, K = W.shape
    nG = K // group_size
    qmax = (1 << bits) - 1
    g = W.reshape(M, nG, group_size)
    wmin = g.min(axis=-1, keepdims=True)
    wmax = g.max(axis=-1, keepdims=True)
    scale = mx.maximum((wmax - wmin) / qmax, 1e-12)
    zero = mx.round(-wmin / scale)
    codes = mx.clip(mx.round(g / scale + zero), 0, qmax).astype(mx.uint32).reshape(M, K)
    scale = scale.reshape(M, nG).astype(mx.float32)
    zero = zero.reshape(M, nG).astype(mx.float32)
    if bits == 4:
        lo = codes[:, 0::2]
        hi = codes[:, 1::2]
        codes = (lo | (hi << 4)).astype(mx.uint8)        # (M, K/2)
    else:
        codes = codes.astype(mx.uint8)                    # (M, K)
    return codes, scale, zero


def dequantize_weight(codes, scale, zero, bits, group_size, M, K):
    nG = K // group_size
    if bits == 4:
        c = codes.astype(mx.uint32)
        full = mx.zeros((M, K), dtype=mx.uint32)
        full[:, 0::2] = c & 0xF
        full[:, 1::2] = (c >> 4) & 0xF
        codes = full
    g = codes.astype(mx.float32).reshape(M, nG, group_size)
    w = scale.reshape(M, nG, 1) * (g - zero.reshape(M, nG, 1))
    return w.reshape(M, K)


def _kernel(K: int, group_size: int, bits: int):
    nG = K // group_size
    if bits == 8:
        fetch = "float code = (float)codes[row*K + i];"
        Kp_decl = ""
    else:  # INT4, packed two-per-byte
        Kp_decl = f"const uint Kp = {K // 2};"
        fetch = ("uint byte = (uint)codes[row*Kp + (i >> 1)];\n"
                 "            float code = (float)((i & 1u) ? (byte >> 4) : (byte & 0xFu));")
    src = f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        const uint K = {K};
        const uint G = {group_size};
        const uint nG = {nG};
        const uint NT = {_THREADS};
        {Kp_decl}
        threadgroup float red[{_THREADS}];

        float local = 0.0f;
        for (uint i = tid; i < K; i += NT) {{
            uint g = i / G;
            float sc = scales[row*nG + g];
            float zp = zeros[row*nG + g];
            {fetch}
            local += sc * (code - zp) * (float)x[i];
        }}
        red[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = NT/2; k > 0; k >>= 1) {{
            if (tid < k) red[tid] += red[tid+k];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) out[row] = red[0];
    """
    return mx.fast.metal_kernel(name=f"qgemv_{bits}_{K}_{group_size}",
                                input_names=["codes", "scales", "zeros", "x"],
                                output_names=["out"], source=src)


def qgemv_metal(codes, scales, zeros, x, bits, group_size, M, K):
    (out,) = _kernel(K, group_size, bits)(
        inputs=[codes, scales, zeros, x.reshape(K).astype(mx.float32)],
        grid=(M * _THREADS, 1, 1), threadgroup=(_THREADS, 1, 1),
        output_shapes=[(M,)], output_dtypes=[mx.float32],
    )
    return out


if __name__ == "__main__":
    mx.random.seed(0)
    M, K, G = 4096, 4096, 64
    W = mx.random.normal((M, K)); x = mx.random.normal((K,)); mx.eval(W, x)
    for bits in (8, 4):
        codes, scale, zero = quantize_weight(W, bits, G)
        mx.eval(codes, scale, zero)
        ref = dequantize_weight(codes, scale, zero, bits, G, M, K) @ x  # kernel target
        ours = qgemv_metal(codes, scale, zero, x, bits, G, M, K)
        mx.eval(ref, ours)
        err = mx.max(mx.abs(ours - ref)).item()
        print(f"INT{bits} qgemv max abs error vs dequant-matmul: {err:.2e}",
              "PASS" if err < 1e-2 else "FAIL")
