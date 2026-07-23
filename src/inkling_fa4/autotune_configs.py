"""BLOCK parameter presets for different GPU architectures and seq lens.

Usage:
    from inkling_fa4.autotune_configs import get_config

    cfg = get_config(seq_len=4096, arch="sm86")
    # cfg == {"BLOCK_Q": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
"""

import torch

# ── config presets ──────────────────────────────────────────────────────────
# Each entry: (min_seq_len, BLOCK_Q, BLOCK_K, num_warps, num_stages)
# Sorted by seq_len ascending; the last entry with seq_len <= current is used.
# "default" is the fallback when no entry matches below the smallest threshold.

PRESETS = {
    # Ampere (RTX 3090, A100, etc.) — 48 KB shared memory
    "sm86": {
        "default": (32, 32, 4, 2),
        "buckets": [
            (512,   32, 64, 4, 2),   # seq <= 512
            (1024,  32, 32, 4, 3),   # seq <= 1024
            (2048,  128, 32, 8, 2),  # seq <= 2048
            (4096,  32, 32, 4, 2),   # seq <= 4096
            # ↑ 以上为 3090 实测最优。大于 4096 用 default
        ],
    },

    # Hopper (H100, H200) — 更高的 shared memory, TMA, wgmma
    # 无实测数据，暂用保守配置；拿到 H100 后重新 autotune 再更新
    "sm90": {
        "default": (64, 64, 8, 4),
        "buckets": [
            (512,   64, 64, 4, 3),
            (1024,  64, 64, 4, 3),
            (2048,  64, 64, 8, 4),
            (4096,  64, 64, 8, 4),
        ],
    },

    # Blackwell (B200, GB200)
    "sm100": {
        "default": (128, 128, 8, 4),
        "buckets": [
            (1024,  128, 64, 8, 4),
            (4096,  128, 128, 8, 4),
        ],
    },
}


def _detect_arch() -> str:
    """Return the compute capability key for the current GPU, e.g. 'sm86'."""
    if not torch.cuda.is_available():
        return "sm86"  # fallback
    cap = torch.cuda.get_device_capability()
    return f"sm{cap[0]}{cap[1]}"


def get_preset(seq_len: int, arch: str | None = None) -> dict:
    """Return the best BLOCK config for a given seq_len and GPU arch.

    Args:
        seq_len: Number of query tokens.
        arch: Compute capability like "sm86", "sm90", or None for auto-detect.

    Returns:
        dict with keys: BLOCK_Q, BLOCK_K, num_warps, num_stages.
    """
    if arch is None:
        arch = _detect_arch()

    arch_presets = PRESETS.get(arch, PRESETS.get("sm86", {}))
    default = arch_presets["default"]
    buckets = arch_presets.get("buckets", [])

    for threshold, bq, bk, nw, ns in buckets:
        if seq_len <= threshold:
            return {"BLOCK_Q": bq, "BLOCK_K": bk,
                    "num_warps": nw, "num_stages": ns}

    bq, bk, nw, ns = default
    return {"BLOCK_Q": bq, "BLOCK_K": bk,
            "num_warps": nw, "num_stages": ns}


# quick self-test
if __name__ == "__main__":
    for seq in [64, 512, 1024, 2048, 4096, 8192]:
        cfg = get_preset(seq, "sm86")
        print(f"sm86 seq={seq:5d}: BQ={cfg['BLOCK_Q']:3d} BK={cfg['BLOCK_K']:3d} "
              f"W={cfg['num_warps']} S={cfg['num_stages']}")

    print()
    for seq in [64, 512, 1024, 4096]:
        cfg = get_preset(seq, "sm90")
        print(f"sm90 seq={seq:5d}: BQ={cfg['BLOCK_Q']:3d} BK={cfg['BLOCK_K']:3d} "
              f"W={cfg['num_warps']} S={cfg['num_stages']}")