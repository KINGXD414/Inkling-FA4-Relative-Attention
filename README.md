# Inkling FA4 Relative Attention — Triton Implementation

Triton implementation of the Inkling FA4 Relative Attention operator, with [distance-based relative positional bias](https://arxiv.org/abs/2603.05451).

## Status

14/14 correctness tests passed on RTX 3090 (sm86) with Triton 3.4.0 / PyTorch 2.8.0.

## Project Structure

```
src/inkling_fa4/
├── __init__.py
├── triton_kernel.py    # Triton FA4 relative attention kernel
└── reference.py        # Pure PyTorch reference (golden)

tests/
└── test_correctness.py # 14 correctness tests

benchmarks/             # Performance benchmarks (WIP)
docs/
├── architecture.md     # Architecture diagram
└── research.md         # Full research document
```

## Quick Start

```bash
pip install -e .
python -m pytest tests/ -v
```

## Reference

- Inkling FA4 in vLLM: `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`
- FlashAttention-4 backend: [vllm-project/tml-fa4](https://github.com/vllm-project/tml-fa4)
- SGLang Inkling: [sgl-project/sglang](https://github.com/sgl-project/sglang)
- FlashAttention-4 paper: <https://arxiv.org/abs/2603.05451>

## License

Apache 2.0