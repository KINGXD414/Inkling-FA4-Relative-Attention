# Inkling FA4 Relative Attention — Triton 实现

基于 tile-wise online softmax 的距离相对位置偏置注意力算子的 Triton 实现。正确性已验证通过（14/14）。

## 性能（RTX 3090, sm86）

| seq_len | 优化前 | 优化后 | 加速比 | 配置（BQ×BK×W×S） |
|---------|--------|--------|--------|-------------------|
| 64 | 0.102ms | 0.103ms | ~1.0x | 32×64×4×2 |
| 128 | 0.094ms | 0.102ms | ~1.0x | 32×64×4×2 |
| 512 | 0.095ms | 0.104ms | ~1.0x | 32×64×4×2 |
| 1024 | 0.193ms | **0.160ms** | **1.21x** | 32×32×4×3 |
| 2048 | 0.500ms | **0.359ms** | **1.39x** | 128×32×8×2 |
| 4096 | 1.473ms | **1.037ms** | **1.42x** | 32×32×4×2 |

## 正确性（14/14）
test_prefill_basic[64/128][4-4/8-2]  ✅  单序列 prefill
test_ragged[4-4/8-2]                 ✅  变长 batch [64,33,17]
test_long_ctx[4-4/8-2]               ✅  长上下文（seq=512 >> rel=128）
test_sliding_window[4-4/8-2]         ✅  滑动窗口（window=63）
test_chunked[4-4/8-2]                ✅  分块 prefill（128 < 256）
test_decode[4-4/8-2]                 ✅  单 token decode

GPU: RTX 3090 | Triton 3.4.0 | PyTorch 2.8.0 | atol=2e-2（与 vllm 一致）

## 快速开始

```bash
pip install -e .
pytest tests/ -v
python benchmarks/benchmark_prefill.py

##配置
BLOCK 参数按 GPU 架构和 seq_len 自动选择（src/inkling_fa4/autotune_configs.py）。当前支持 sm86（Ampere），sm90（Hopper）/ sm100（Blackwell）预设已预留，拿到卡后需重新 autotune。

##开源实现参考
vllm inkling
tml-fa4 — FA4 底层 kernel
sglang inkling
tokenspeed inkling
huggingface inkling
FA4 论文

##项目结构
src/inkling_fa4/
├── triton_kernel.py          # 主 kernel
├── autotune_configs.py       # 架构感知的 BLOCK 配置
├── reference.py              # PyTorch 参考实现（不改）
tests/
├── test_correctness.py       # 14 个正确性测试
benchmarks/
├── benchmark_prefill.py      # 性能基准
├── autotune.py               # BLOCK 参数自动搜索
docs/
├── workflow.md               # 协作流程

##License
Apache 2.0

```