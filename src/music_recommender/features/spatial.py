"""Stereo / spatial analysis on the ORIGINAL channel pair.

`preprocess` used to downmix to mono (`-ac 1`) before anything saw the file,
which made every spatial claim unfalsifiable — the information was gone. This
module runs on the ORIGINAL stereo decode, while the existing mono path stays
untouched for the legacy algorithms.

All outputs are measurable proxies, nothing more:

  - lr_balance_db: L/R level difference. |balance| large is not "wider", just
    panned to one side.
  - lr_correlation: -1..1. Near +1 on real music often means dual-mono content,
    NOT a missing stereo file; near 0 is uncorrelated (wide), near -1 is
    anti-phase (which can blow up side energy without sounding "wide" at all).
  - side_energy_ratio: E(S)/(E(M)+E(S)) with M=(L+R)/2, S=(L-R)/2. Must always
    be read TOGETHER with correlation. Width, spectral width and reverb depth
    are three different dimensions; none of them is derivable from this module.

Mono files report status "unavailable/mono" — a real dual-mono file returns
high correlation and near-zero side ratio, which is a measurement, not missing
data, and the two must stay distinguishable.
"""
from __future__ import annotations

import numpy as np

DEFAULTS: dict = {
    "curve_interval_seconds": 0.5,
    # below this RMS the channel pair is metering noise: correlation/division
    # results would be garbage, so those buckets are marked invalid (None).
    "min_frame_rms": 1e-4,
    "frame_length": 2048,
    "hop_length": 512,
}


def _cfg(cfg: dict | None, key: str):
    if cfg:
        v = cfg.get("analysis", {}).get(key)
        if v is not None:
            return v
    return DEFAULTS[key]


def spatial_curves(stereo: np.ndarray, sr: int, cfg: dict | None = None) -> dict:
    """Bucket-grid spatial curves from an (2, n) or (n,) array of original channels."""
    interval = float(_cfg(cfg, "curve_interval_seconds"))
    min_rms = float(_cfg(cfg, "min_frame_rms"))
    fl = int(_cfg(cfg, "frame_length"))
    hop = int(_cfg(cfg, "hop_length"))

    unavailable = lambda reason: {"status": "unavailable", "reason": reason,
                                  "curves": {}, "valid": [], "meta": {}}
    if stereo.ndim == 1 or (stereo.ndim == 2 and stereo.shape[0] < 2):
        return unavailable("mono")
    if stereo.shape[0] > 2:
        # multi-channel files must NOT silently read "front two = the mix"; the
        # downmix policy is a decision, so refuse rather than guess.
        return unavailable(f"multichannel_{stereo.shape[0]}_no_downmix_policy")

    l = np.asarray(stereo[0], np.float64)
    r = np.asarray(stereo[1], np.float64)
    n = min(l.size, r.size)
    l, r = l[:n], r[:n]
    if n < fl:
        return unavailable("too_short")

    duration = n / sr
    n_buckets = max(1, int(np.ceil(duration / interval)))
    bucket_times = [round((i + 0.5) * interval, 3) for i in range(n_buckets)]
    step = max(1, int(round(0.25 * sr / hop)))  # a spatial frame every ~0.25 s

    acc: dict[int, dict[str, list[float]]] = {}
    valid = [False] * n_buckets
    starts = range(0, n - fl, hop * step)
    for start in starts:
        wl = l[start:start + fl]
        wr = r[start:start + fl]
        if float(np.sqrt((wl ** 2 + wr ** 2).mean() / 2.0)) < min_rms:
            continue
        bi = min(n_buckets - 1, int((start + fl / 2) / sr / interval))
        lc = wl - wl.mean()
        rc = wr - wr.mean()
        denom = float(np.sqrt((lc ** 2).sum()) * np.sqrt((rc ** 2).sum()))
        m = (wl + wr) / 2.0
        s = (wl - wr) / 2.0
        em, es = float((m ** 2).sum()), float((s ** 2).sum())
        slot = acc.setdefault(bi, {"bal": [], "corr": [], "side": []})
        slot["bal"].append(20.0 * np.log10((np.sqrt(np.mean(wl ** 2)) + 1e-9)
                                           / (np.sqrt(np.mean(wr ** 2)) + 1e-9)))
        slot["corr"].append(float((lc * rc).sum() / denom) if denom > 0 else 0.0)
        slot["side"].append(es / (em + es) if (em + es) > 0 else 0.0)
        valid[bi] = True

    def per_bucket(key: str) -> list:
        return [round(float(np.mean(acc[i][key])), 4) if valid[i] and acc.get(i) else None
                for i in range(n_buckets)]

    return {
        "status": "ready",
        "reason": "",
        "time_seconds": bucket_times,
        "curves": {
            "lr_balance_db": per_bucket("bal"),
            "lr_correlation": per_bucket("corr"),
            "side_energy_ratio": per_bucket("side"),
        },
        "valid": valid,
        "meta": {
            "scope": "full_track",
            "interval_seconds": interval,
            "n_points": n_buckets,
            "time_semantics": "bucket_center",
            "source": "original_stereo",
            "method_version": "spatial-v1",
            "notes": [
                "side_energy_ratio 必须与 lr_correlation 一起读：反相会让 side 比例异常大，但听起来并不宽",
                "宽度、频谱宽度、混响深度是三个不同维度；本模块只测前者的左右代理量",
                "lr_correlation 接近 +1 的真实音乐常是双单声道内容，不是缺了立体声",
                "invalid 桶（None）是电平低于计量底噪，不是完美居中",
            ],
        },
    }
