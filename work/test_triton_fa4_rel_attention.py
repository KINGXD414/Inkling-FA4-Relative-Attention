import torch
import pytest
from triton_fa4_rel_attention import inkling_fa4_rel_attention_triton

HEAD_DIM = 128
REL_EXTENT = 1024
HEAD_CONFIGS = [(4, 4), (8, 2)]
DTYPE = torch.bfloat16

def ref_rel_attn(q, k, v, rel_logits, cu_seqlens_q, cu_seqlens_k, scale, rel_extent, causal=True, window_left=None):
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

def make_case(batch, sql_q, sql_k, n_heads, kv_heads, rel_ext=REL_EXTENT):
    tq, tk = sum(sql_q), sum(sql_k)
    q = torch.randn(tq, n_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(tk, kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(tk, kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    rl = torch.randn(tq, n_heads, rel_ext, dtype=DTYPE, device="cuda")
    cuq = torch.zeros(batch+1, dtype=torch.int32, device="cuda")
    cuk = torch.zeros(batch+1, dtype=torch.int32, device="cuda")
    for i in range(batch): cuq[i+1] = cuq[i] + sql_q[i]; cuk[i+1] = cuk[i] + sql_k[i]
    return q, k, v, rl, cuq, cuk

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
@pytest.mark.parametrize("seq_len", [64, 128])
def test_prefill_basic(n_heads, kv_heads, seq_len):
    q,k,v,rl,cuq,cuk = make_case(1,[seq_len],[seq_len],n_heads,kv_heads)
    torch.testing.assert_close(inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,seq_len,seq_len,REL_EXTENT).float(), ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(), atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_ragged(n_heads, kv_heads):
    sql=[64,33,17]; mx=max(sql)
    q,k,v,rl,cuq,cuk = make_case(3,sql,sql,n_heads,kv_heads)
    torch.testing.assert_close(inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,mx,mx,REL_EXTENT).float(), ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(), atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_long_ctx(n_heads, kv_heads):
    sl,re=512,128
    q,k,v,rl,cuq,cuk = make_case(1,[sl],[sl],n_heads,kv_heads,rel_ext=re)
    rl=rl[:,:,:re].contiguous()
    torch.testing.assert_close(inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,sl,sl,re).float(), ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,re).float(), atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_sliding_window(n_heads, kv_heads):
    sl,le=128,64
    q,k,v,rl,cuq,cuk = make_case(1,[sl],[sl],n_heads,kv_heads,rel_ext=le)
    rl=rl[:,:,:le].contiguous()
    torch.testing.assert_close(inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,sl,sl,le,window_size=(le-1,0)).float(), ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,le,window_left=le-1).float(), atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_chunked(n_heads, kv_heads):
    ql,kl=128,256
    q,k,v,rl,cuq,cuk = make_case(1,[ql],[kl],n_heads,kv_heads)
    torch.testing.assert_close(inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,ql,kl,REL_EXTENT).float(), ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(), atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_decode(n_heads, kv_heads):
    kl=128
    q,k,v,rl,cuq,cuk = make_case(1,[1],[kl],n_heads,kv_heads)
    torch.testing.assert_close(inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,1,kl,REL_EXTENT).float(), ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(), atol=2e-2, rtol=2e-2)