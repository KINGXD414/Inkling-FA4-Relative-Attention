"""Decode-specialized Triton kernel for Inkling FA4 Relative Attention.

Unlike the prefill kernel (triton_kernel.py), this kernel processes a SINGLE
query token rather than a BLOCK_Q tile. This eliminates the ~97% waste from
loading/computing 31 masked rows when q_len = 1.

Changes from prefill kernel:
- Single Q vector instead of [BLOCK_Q, head_dim] matrix
- m_i / l_i are scalars instead of [BLOCK_Q, 1] matrices
- acc is [head_dim] vector instead of [BLOCK_Q, head_dim] matrix
- Score: tl.sum(q * k, axis=1) instead of tl.dot(q, k)
- V accum: tl.sum(p[:, None] * v, axis=0) instead of tl.dot(p, v)
- Grid: (n_heads,) without Q tile dimension
"""

import torch
import triton
import triton.language as tl
from inkling_fa4 import autotune_configs


@triton.jit
def _fa4_rel_attn_decode_kernel(
    q_ptr, k_ptr, v_ptr, rel_ptr, out_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    stride_q_t, stride_q_h,
    stride_k_t, stride_k_h,
    stride_v_t, stride_v_h,
    stride_rel_t, stride_rel_h,
    stride_out_t, stride_out_h,
    seq_id, n_heads, kv_heads,
    head_dim: tl.constexpr,
    rel_extent: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
):
    # --- sequence bounds ---
    q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    k_start = tl.load(cu_seqlens_k_ptr + seq_id)
    k_end = tl.load(cu_seqlens_k_ptr + seq_id + 1)
    k_len = k_end - k_start

    h = tl.program_id(0)
    kv_h = h // (n_heads // kv_heads)

    off_d = tl.arange(0, head_dim)
    off_k = tl.arange(0, BLOCK_K)

    # --- load single Q vector ---
    q = tl.load(
        q_ptr + q_start * stride_q_t + h * stride_q_h + off_d,
        mask=off_d < head_dim, other=0.0
    ).to(tl.float16)

    scale = 1.0 / tl.sqrt(tl.full([1], head_dim, dtype=tl.float16))

    # --- online softmax (single-row) ---
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    offset_seq = (q_end - q_start) - (k_end - k_start)

    for start_k in range(0, k_len, BLOCK_K):
        k_tile_sz = min(BLOCK_K, k_len - start_k)
        k_idx = k_start + start_k + off_k

        # causal + sliding window mask (single row)
        mask = tl.full([BLOCK_K], 1, dtype=tl.int1)
        if CAUSAL:
            mask = mask & ((start_k + off_k) <= (q_start - k_start))
        if WINDOW_LEFT >= 0:
            mask = mask & (((q_start - k_start) - (start_k + off_k)) <= WINDOW_LEFT)

        # --- load K tile ---
        k = tl.load(
            k_ptr + k_idx[:, None] * stride_k_t + kv_h * stride_k_h + off_d[None, :],
            mask=(off_k[:, None] < k_tile_sz) & (off_d[None, :] < head_dim),
            other=0.0
        ).to(tl.float16)

        # --- QK^T scores (single query, K tile) ---
        s = tl.sum(q[None, :] * k, axis=1) * scale

        # --- relative bias ---
        rel_dist = (q_start + 0 + offset_seq) - k_idx
        in_range = (rel_dist >= 0) & (rel_dist < rel_extent)
        rel_idx = tl.where(in_range, rel_dist, tl.zeros_like(rel_dist))
        rp = rel_ptr + q_start * stride_rel_t + h * stride_rel_h + rel_idx
        rv = tl.load(rp, mask=in_range & (off_k < k_tile_sz), other=0.0).to(tl.float32)
        s = s + tl.where(in_range & mask, rv, 0.0)

        # --- causal + window mask ---
        s = tl.where(mask, s, -float("inf"))

        # --- online softmax (single row) ---
        m_ij = tl.max(s)
        all_inf = m_ij <= -1e8
        m_ij_safe = tl.where(all_inf, tl.zeros_like(m_ij), m_ij)
        m_new = tl.maximum(m_i, m_ij_safe)
        p = tl.exp(s - m_new)
        l_ij = tl.sum(p)

        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha
        l_i = l_i * alpha

        # --- V weighted sum ---
        v = tl.load(
            v_ptr + k_idx[:, None] * stride_v_t + kv_h * stride_v_h + off_d[None, :],
            mask=(off_k[:, None] < k_tile_sz) & (off_d[None, :] < head_dim),
            other=0.0
        ).to(tl.float16)
        acc = acc + tl.sum(p[:, None].to(tl.float16) * v, axis=0).to(tl.float32)
        l_i = l_i + l_ij
        m_i = m_new

    acc = acc / (l_i + 1e-30)
    tl.store(
        out_ptr + q_start * stride_out_t + h * stride_out_h + off_d,
        acc.to(tl.bfloat16), mask=off_d < head_dim
    )


@torch.no_grad()
def inkling_fa4_rel_attention_decode_triton(
    q, k, v, rel_logits,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    rel_extent,
    causal=True,
    window_size=(-1, -1),
    n_heads=None, kv_heads=None,
    head_dim=128,
    arch=None,
):
    """Decode-only Triton FA4 relative attention.

    Each sequence contributes exactly 1 query token. For ragged batches,
    each sequence's query token attends to its own KV cache segment.
    """
    if n_heads is None:
        n_heads = q.shape[1]
    if kv_heads is None:
        kv_heads = k.shape[1]
    out = torch.empty_like(q)
    batch = cu_seqlens_q.shape[0] - 1

    cfg = autotune_configs.get_preset(1, arch=arch)
    BLOCK_K = cfg["BLOCK_K"]

    for seq_id in range(batch):
        grid = (n_heads,)
        _fa4_rel_attn_decode_kernel[grid](
            q, k, v, rel_logits, out,
            cu_seqlens_q, cu_seqlens_k,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            rel_logits.stride(0), rel_logits.stride(1),
            out.stride(0), out.stride(1),
            seq_id, n_heads, kv_heads,
            head_dim, rel_extent, BLOCK_K,
            CAUSAL=causal, WINDOW_LEFT=window_size[0],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out


# --- test: verify decode matches prefill kernel ---
if __name__ == "__main__":
    torch.manual_seed(42)
    from inkling_fa4.triton_kernel import inkling_fa4_rel_attention_triton

    n_heads, kv_heads, head_dim = 4, 4, 128
    kv_len = 128
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(1, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, kv_heads, head_dim, dtype=dtype, device=device)
    rel = torch.randn(1, n_heads, 1024, dtype=dtype, device=device)
    cuq = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cuk = torch.tensor([0, kv_len], dtype=torch.int32, device=device)

    out_prefill = inkling_fa4_rel_attention_triton(
        q, k, v, rel, cuq, cuk, 1, kv_len, 1024
    )
    out_decode = inkling_fa4_rel_attention_decode_triton(
        q, k, v, rel, cuq, cuk, 1, kv_len, 1024
    )

    diff = (out_decode.float() - out_prefill.float()).abs()
    print(f"Decode kernel vs prefill kernel: max_diff={diff.max():.6f}")
    if diff.max() <= 2e-2:
        print("MATCH (within tolerance)")
    else:
        print("MISMATCH")