"""Triton prefill kernel with causal/sliding-window KV range pruning.

This module is an optimization branch of ``triton_kernel.py``.  The original
kernel is intentionally left unchanged so that it remains available as the
baseline for correctness and performance comparisons.

The optimization in this file is deliberately narrow:

* Causal attention skips Key tiles that are entirely in the future.
* Sliding-window attention skips Key tiles entirely left of the window.
* Element-wise masks are retained for diagonal, window-boundary, and tail
  tiles, preserving the baseline kernel's current attention semantics.

No changes are made here to the relative-position convention, online softmax,
tensor-core dot products, output dtype, or autotune presets.
"""

import torch
import triton
import triton.language as tl

from inkling_fa4 import autotune_configs


@triton.jit
def _fa4_rel_attn_prefill_kv_pruned_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    rel_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    stride_q_t,
    stride_q_h,
    stride_k_t,
    stride_k_h,
    stride_v_t,
    stride_v_h,
    stride_rel_t,
    stride_rel_h,
    stride_out_t,
    stride_out_h,
    seq_id,
    n_heads,
    kv_heads,
    head_dim: tl.constexpr,
    rel_extent: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
):
    q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    k_start = tl.load(cu_seqlens_k_ptr + seq_id)
    k_end = tl.load(cu_seqlens_k_ptr + seq_id + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    h = tl.program_id(0)
    pid_q = tl.program_id(1)
    kv_h = h // (n_heads // kv_heads)

    q_off = pid_q * BLOCK_Q
    q_tile_sz = min(BLOCK_Q, q_len - q_off)
    if q_tile_sz <= 0:
        return

    off_q = tl.arange(0, BLOCK_Q)
    off_d = tl.arange(0, head_dim)
    off_k = tl.arange(0, BLOCK_K)

    q_idx = q_start + q_off + off_q
    q_ptrs = (
        q_ptr
        + q_idx[:, None] * stride_q_t
        + h * stride_q_h
        + off_d[None, :]
    )
    q_mask = (off_q[:, None] < q_tile_sz) & (
        off_d[None, :] < head_dim
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_Q, 1], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q, 1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, head_dim], dtype=tl.float32)

    # Keep the baseline kernel's relative-position convention unchanged.
    offset_seq = q_len - k_len

    # Compute the union of Key positions that may be visible to any Query in
    # this Query tile.  Bounds are expressed in sequence-local coordinates.
    #
    # Current baseline causal semantics:
    #     key_position <= query_position
    # Therefore the largest useful Key is the last valid Query in the tile.
    first_k = 0
    last_k_exclusive = k_len

    if CAUSAL:
        q_tile_end_exclusive = q_off + q_tile_sz
        last_k_exclusive = min(last_k_exclusive, q_tile_end_exclusive)

    if WINDOW_LEFT >= 0:
        first_visible_k = q_off - WINDOW_LEFT
        first_visible_k = max(first_visible_k, 0)
        # Align down because the first retained block may contain a mixture of
        # visible and masked Key positions.
        first_k = (first_visible_k // BLOCK_K) * BLOCK_K

    # Blocks outside [first_k, last_k_exclusive) are entirely masked by the
    # baseline causal/window rules, so their K/V loads, relative-bias loads,
    # QK dot products, softmax work, and PV dot products can all be skipped.
    for start_k in range(first_k, last_k_exclusive, BLOCK_K):
        k_tile_sz = min(BLOCK_K, k_len - start_k)
        k_idx = k_start + start_k + off_k

        # Retain element-wise masks for diagonal/window-boundary/tail blocks.
        mask = tl.full([BLOCK_Q, BLOCK_K], 1, dtype=tl.int1)
        if CAUSAL:
            mask = mask & (
                (q_off + off_q[:, None])
                >= (start_k + off_k[None, :])
            )
        if WINDOW_LEFT >= 0:
            mask = mask & (
                (q_off + off_q[:, None])
                - (start_k + off_k[None, :])
                <= WINDOW_LEFT
            )

        k_ptrs = (
            k_ptr
            + k_idx[None, :] * stride_k_t
            + kv_h * stride_k_h
            + off_d[:, None]
        )
        k_mask = (off_k[None, :] < k_tile_sz) & (
            off_d[:, None] < head_dim
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float16)
        s = tl.dot(q, k).to(tl.float32) * (1.0 / head_dim)

        rel_dist = (
            q_off
            + off_q[:, None]
            + offset_seq
            - (start_k + off_k[None, :])
        )
        in_range = (rel_dist >= 0) & (rel_dist < rel_extent)
        rel_idx = tl.where(in_range, rel_dist, tl.zeros_like(rel_dist))
        rel_ptrs = (
            rel_ptr
            + (q_start + q_off + off_q[:, None]) * stride_rel_t
            + h * stride_rel_h
            + rel_idx
        )
        rel_mask = (
            in_range
            & (off_q[:, None] < q_tile_sz)
            & (off_k[None, :] < k_tile_sz)
        )
        rel_value = tl.load(
            rel_ptrs, mask=rel_mask, other=0.0
        ).to(tl.float32)

        s = tl.where(in_range & mask, s + rel_value, s)
        s = tl.where(mask, s, -float("inf"))

        m_ij = tl.max(s, axis=1)[:, None]
        all_inf = m_ij <= -1e8
        m_ij_safe = tl.where(all_inf, tl.zeros_like(m_ij), m_ij)
        m_new = tl.maximum(m_i, m_ij_safe)
        p = tl.exp(s - m_new)
        l_ij = tl.sum(p, axis=1)[:, None]
        alpha = tl.exp(m_i - m_new)

        acc = acc * alpha
        l_i = l_i * alpha

        v_ptrs = (
            v_ptr
            + k_idx[:, None] * stride_v_t
            + kv_h * stride_v_h
            + off_d[None, :]
        )
        v_mask = (off_k[:, None] < k_tile_sz) & (
            off_d[None, :] < head_dim
        )
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float16)
        acc = acc + tl.dot(p.to(tl.float16), v).to(tl.float32)
        l_i = l_i + l_ij
        m_i = m_new

    acc = acc / (l_i + 1e-30)
    out_ptrs = (
        out_ptr
        + q_idx[:, None] * stride_out_t
        + h * stride_out_h
        + off_d[None, :]
    )
    out_mask = (off_q[:, None] < q_tile_sz) & (
        off_d[None, :] < head_dim
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@torch.no_grad()
def inkling_fa4_rel_attention_kv_pruned_triton(
    q,
    k,
    v,
    rel_logits,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    rel_extent,
    causal=True,
    window_size=(-1, -1),
    n_heads=None,
    kv_heads=None,
    head_dim=128,
    arch=None,
):
    """Run the KV-range-pruned Inkling relative-attention prefill kernel.

    The signature intentionally matches ``inkling_fa4_rel_attention_triton``
    except for the function name, making baseline/optimized A/B comparisons
    possible without changing call arguments.
    """
    if n_heads is None:
        n_heads = q.shape[1]
    if kv_heads is None:
        kv_heads = k.shape[1]

    out = torch.empty_like(q)
    batch = cu_seqlens_q.shape[0] - 1

    for seq_id in range(batch):
        q_len = int(
            cu_seqlens_q[seq_id + 1] - cu_seqlens_q[seq_id]
        )
        cfg = autotune_configs.get_preset(q_len, arch=arch)
        grid = (n_heads, triton.cdiv(q_len, cfg["BLOCK_Q"]))
        _fa4_rel_attn_prefill_kv_pruned_kernel[grid](
            q,
            k,
            v,
            rel_logits,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            rel_logits.stride(0),
            rel_logits.stride(1),
            out.stride(0),
            out.stride(1),
            seq_id,
            n_heads,
            kv_heads,
            head_dim,
            rel_extent,
            cfg["BLOCK_Q"],
            cfg["BLOCK_K"],
            CAUSAL=causal,
            WINDOW_LEFT=window_size[0],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return out


__all__ = [
    "inkling_fa4_rel_attention_kv_pruned_triton",
]
