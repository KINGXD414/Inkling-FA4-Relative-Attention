"""Benchmark script for Inkling FA4 Relative Attention (Triton).

Usage:
    python benchmarks/benchmark_prefill.py
    python benchmarks/benchmark_prefill.py --seq-len 512 --verbose

Output: latency (ms), TFLOPs/s, and BLOCK config used.
"""

import argparse
import torch
from inkling_fa4 import inkling_fa4_rel_attention_triton, autotune_configs

DTYPE = torch.bfloat16
HEAD_DIM = 128
WARMUP = 10
REPEAT = 50


def benchmark_prefill(seq_len, n_heads, kv_heads, causal=True, verbose=False):
    total_q = seq_len
    total_k = seq_len
    q = torch.randn(total_q, n_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(total_k, kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(total_k, kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    rel = torch.randn(total_q, n_heads, 1024, dtype=DTYPE, device="cuda")
    cuq = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cuk = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    if verbose:
        cfg = autotune_configs.get_preset(seq_len)
        print(f"  [config] BQ={cfg['BLOCK_Q']} BK={cfg['BLOCK_K']} "
              f"W={cfg['num_warps']} S={cfg['num_stages']}")

    for _ in range(WARMUP):
        inkling_fa4_rel_attention_triton(q, k, v, rel, cuq, cuk, seq_len, seq_len, 1024)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(REPEAT):
        inkling_fa4_rel_attention_triton(q, k, v, rel, cuq, cuk, seq_len, seq_len, 1024)
    end_event.record()
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / REPEAT

    flops = 2 * seq_len * seq_len * HEAD_DIM * n_heads
    tflops = flops / ms / 1e9
    return ms, tflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    arch = autotune_configs._detect_arch()
    print(f"GPU: {torch.cuda.get_device_name(0)} ({arch})")
    if args.verbose:
        print()

    seq_lens = [args.seq_len] if args.seq_len else [64, 128, 256, 512, 1024, 2048, 4096]

    print(f"{'seq_len':>8} {'n_heads':>8} {'kv_heads':>8} {'latency(ms)':>12} {'TFLOPs/s':>10}")
    print("-" * 50)
    for sl in seq_lens:
        try:
            ms, tflops = benchmark_prefill(sl, args.heads, args.kv_heads, verbose=args.verbose)
            print(f"{sl:>8} {args.heads:>8} {args.kv_heads:>8} {ms:>12.3f} {tflops:>10.2f}")
        except Exception as e:
            print(f"{sl:>8} {args.heads:>8} {args.kv_heads:>8} {'OOM/ERR':>12}")


if __name__ == "__main__":
    main()