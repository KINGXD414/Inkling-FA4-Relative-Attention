import torch
import pytest
from inkling_fa4 import inkling_fa4_rel_attention_triton
from inkling_fa4.reference import ref_rel_attn

HEAD_DIM = 128
REL_EXTENT = 1024
HEAD_CONFIGS = [(4, 4), (8, 2)]
DTYPE = torch.bfloat16

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
    torch.testing.assert_close(
        inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,seq_len,seq_len,REL_EXTENT).float(),
        ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(),
        atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_ragged(n_heads, kv_heads):
    sql=[64,33,17]; mx=max(sql)
    q,k,v,rl,cuq,cuk = make_case(3,sql,sql,n_heads,kv_heads)
    torch.testing.assert_close(
        inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,mx,mx,REL_EXTENT).float(),
        ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(),
        atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_long_ctx(n_heads, kv_heads):
    sl,re=512,128
    q,k,v,rl,cuq,cuk = make_case(1,[sl],[sl],n_heads,kv_heads,rel_ext=re)
    rl=rl[:,:,:re].contiguous()
    torch.testing.assert_close(
        inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,sl,sl,re).float(),
        ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,re).float(),
        atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_sliding_window(n_heads, kv_heads):
    sl,le=128,64
    q,k,v,rl,cuq,cuk = make_case(1,[sl],[sl],n_heads,kv_heads,rel_ext=le)
    rl=rl[:,:,:le].contiguous()
    torch.testing.assert_close(
        inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,sl,sl,le,window_size=(le-1,0)).float(),
        ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,le,window_left=le-1).float(),
        atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_chunked(n_heads, kv_heads):
    ql,kl=128,256
    q,k,v,rl,cuq,cuk = make_case(1,[ql],[kl],n_heads,kv_heads)
    torch.testing.assert_close(
        inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,ql,kl,REL_EXTENT).float(),
        ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(),
        atol=2e-2, rtol=2e-2)

@pytest.mark.parametrize("n_heads,kv_heads", HEAD_CONFIGS)
def test_decode(n_heads, kv_heads):
    kl=128
    q,k,v,rl,cuq,cuk = make_case(1,[1],[kl],n_heads,kv_heads)
    torch.testing.assert_close(
        inkling_fa4_rel_attention_triton(q,k,v,rl,cuq,cuk,1,kl,REL_EXTENT).float(),
        ref_rel_attn(q,k,v,rl,cuq,cuk,1.0/HEAD_DIM,REL_EXTENT).float(),
        atol=2e-2, rtol=2e-2)