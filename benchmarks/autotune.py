"""Autotune BLOCK parameters for Inkling FA4 Relative Attention.

Grid search over BLOCK_Q, BLOCK_K, num_warps, num_stages
to find the fastest configuration on the target GPU.

Results are saved to outputs/autotune_results.csv
"""

import csv
import itertools
import torch
import triton
from inkling_fa4 import inkling_fa4_rel_attention_triton

DTYPE = torch.bfloat16
HEAD_DIM = 128
WARMUP = 5
REPEAT = 20

# ---- search space ----
BLOCK_QS = [16, 32, 64, 128]
BLOCK_KS = [32, 64, 128]
WARPS = [4, 8]
STAGES = [2, 3, 4]

SEQ_LENS = [512, 1024, 2048, 4096]
HEADS_CFG = [(4, 4), (8, 2)]


def benchmark_config(seq_len, n_heads, kv_heads, BQ, BK, nw, ns):
    """Build a custom kernel with given params and measure latency."""
    total_q = seq_len
    total_k = seq_len
    q = torch.randn(total_q, n_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(total_k, kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(total_k, kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    rel = torch.randn(total_q, n_heads, 1024, dtype=DTYPE, device="cuda")
    cuq = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cuk = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    # We need to override the kernel's BLOCK constants.
    # Approach: compile with the desired config by creating a wrapper
    # that passes BLOCK_Q and BLOCK_K as constexpr arguments.
    # Since inkling_fa4_rel_attention_triton hardcodes 32/64,
    # we directly call the kernel with our parameters.

    from inkling_fa4.triton_kernel import _fa4_rel_attn_prefill_kernel

    # warmup
    for _ in range(WARMUP):
        _run_kernel(q, k, v, rel, cuq, cuk, seq_len, n_heads, kv_heads,
                    BQ, BK, nw, ns, _fa4_rel_attn_prefill_kernel)

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        _run_kernel(q, k, v, rel, cuq, cuk, seq_len, n_heads, kv_heads,
                    BQ, BK, nw, ns, _fa4_rel_attn_prefill_kernel)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / REPEAT

    flops = 2 * seq_len * seq_len * HEAD_DIM * n_heads
    tflops = flops / ms / 1e9
    return ms, tflops


def _run_kernel(q, k, v, rel, cuq, cuk, seq_len, n_heads, kv_heads,
                BQ, BK, nw, ns, kernel_fn):
    """Helper: launch kernel with custom BLOCK config and num_warps/num_stages."""
    grid = (n_heads, triton.cdiv(seq_len, BQ))
    kernel_fn[grid](q, k, v, rel, None,
                    cuq, cuk,
                    q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                    v.stride(0), v.stride(1), rel.stride(0), rel.stride(1),
                    0, 0,  # out strides (dummy for None)
                    0, n_heads, kv_heads, HEAD_DIM, 1024, BQ, BK,
                    CAUSAL=True, WINDOW_LEFT=-1,
                    num_warps=nw, num_stages=ns)


def main():
    results = []
    seen = set()

    for n_heads, kv_heads in HEADS_CFG:
        for sl in SEQ_LENS:
            best_ms = float("inf")
            best_cfg = None
            for BQ in BLOCK_QS:
                for BK in BLOCK_KS:
                    for nw in WARPS:
                        for ns in STAGES:
                            cfg_key = (sl, n_heads, BQ, BK, nw, ns)
                            if cfg_key in seen:
                                continue
                            seen.add(cfg_key)
                            try:
                                ms, tflops = benchmark_config(
                                    sl, n_heads, kv_heads, BQ, BK, nw, ns)
                                results.append(
                                    (sl, n_heads, kv_heads, BQ, BK, nw, ns, ms, tflops))
                                print(
                                    f"  H{n_heads}KV{kv_heads} L{sl} "
                                    f"BQ={BQ} BK={BK} W={nw} S={ns}: "
                                    f"{ms:.3f}ms {tflops:.1f}T"
                                )
                                if ms < best_ms:
                                    best_ms = ms
                                    best_cfg = (BQ, BK, nw, ns)
                            except Exception as e:
                                print(
                                    f"  H{n_heads}KV{kv_heads} L{sl} "
                                    f"BQ={BQ} BK={BK} W={nw} S={ns}: ERR {e}")

            if best_cfg:
                print(f"  >> BEST: H{n_heads}KV{kv_heads} L{sl}: "
                      f"BQ={best_cfg[0]} BK={best_cfg[1]} "
                      f"W={best_cfg[2]} S={best_cfg[3]} = {best_ms:.3f}ms")

    # save
    import os
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "autotune_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_len", "n_heads", "kv_heads",
                     "BLOCK_Q", "BLOCK_K", "warps", "stages",
                     "latency_ms", "tflops"])
        for r in results:
            w.writerow(r)

    print(f"\nResults saved to {path}")
    print("Done!")


if __name__ == "__main__":
    main()