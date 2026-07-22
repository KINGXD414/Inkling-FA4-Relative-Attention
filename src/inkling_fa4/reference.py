"""Pure PyTorch reference implementation of Inkling FA4 Relative Attention."""
import torch

def ref_rel_attn(q, k, v, rel_logits, cu_seqlens_q, cu_seqlens_k, scale, rel_extent, causal=True, window_left=None):
    """Per-position reference (golden) for Inkling relative attention.

    Args:
        q: [total_q, n_heads, head_dim] BF16
        k: [total_q, kv_heads, head_dim] BF16
        v: [total_q, kv_heads, head_dim] BF16
        rel_logits: [total_q, n_heads, rel_extent] BF16
        scale: float (1/head_dim)
        causal: bool
        window_left: int or None for sliding window

    Returns:
        out: [total_q, n_heads, head_dim]
    """
    n_heads = q.shape[1]; kv_heads = k.shape[1]; batch = cu_seqlens_q.shape[0] - 1
    g = n_heads // kv_heads; out = torch.zeros_like(q)
    for b in range(batch):
        qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b+1])
        ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b+1])
        q_b, rl_b = q[qs:qe].float(), rel_logits[qs:qe].float()
        for h in range(n_heads):
            kh = h // g
            scores = torch.einsum("qd,kd->qk", q_b[:, h], k[ks:ke, kh].float()) * scale
            offset = (qe - qs) - (ke - ks)
            q_pos = torch.arange(qe-qs, device=q.device).view(-1,1)
            k_pos = torch.arange(ke-ks, device=k.device).view(1,-1)
            rel_dist = (q_pos + offset) - k_pos
            in_range = (rel_dist >= 0) & (rel_dist < rel_extent)
            bias = torch.gather(rl_b[:, h], 1, rel_dist.clamp(0, rel_extent-1))
            bias[~in_range] = 0.0
            scores = scores + bias
            if causal: scores[~(q_pos >= k_pos)] = -float("inf")
            if window_left is not None: scores[~((q_pos - k_pos) <= window_left)] = -float("inf")
            p = torch.softmax(scores.to(torch.float32), dim=-1)
            out[qs:qe, h] = torch.einsum("qk,kd->qd", p, v[ks:ke, kh].float()).to(out.dtype)
    return out