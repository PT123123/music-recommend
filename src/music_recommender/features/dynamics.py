"""Full-track dynamics: time-stamped curves plus build / peak / pull-back / release events.

Why this module exists (docs/MUSIC_ANALYSIS_REFACTOR_PLAN.md 2.2): the extractor used to
hand the loudest 45 s slice to the structure analysis, so `energy_curve` described that
slice, not the song. Intro, pre-chorus, bridge and outro were systematically missing and
any "where is the climax" claim was biased. Everything here runs on the WHOLE decoded
track on purpose, and only cheap frame-level DSP: no HPSS, no pitch tracking, so the cost
stays a fraction of the expensive window features in `extract.py`.

Honesty rules (spec principle 1):
  - `onset_activity` is an onset-envelope proxy for percussive activity, NOT a drum track.
  - `low_band_energy` is a 20-250 Hz band *share*, NOT a bass-instrument count.
  - `confidence` on events is a model score in [0,1], not a probability; it becomes a
    probability only after calibration against human labels (see `confidence_kind`).
  - Curves keep two scales: `rms_db` is absolute dBFS and comparable across tracks,
    `rms_norm` is within-track 0..1. Per-track normalisation alone cannot express that a
    quiet folk recording is quieter than a mastered pop track.
"""
from __future__ import annotations

import numpy as np

DEFAULTS: dict = {
    "frame_length": 2048,
    "hop_length": 512,
    "curve_interval_seconds": 0.5,
    "low_band_hz": [20.0, 250.0],
    "smooth_seconds": 4.0,
    "min_dynamic_range_db": 1.5,
    # dBFS floor: below -80 dBFS a frame is metering noise or digital silence, and letting it
    # read -240 dB turned the STFT edge frames into a fake 240 dB "dynamic range".
    "rms_db_floor": -80.0,
    "peak_prominence_fraction": 0.35,
    "peak_min_distance_seconds": 10.0,
    "max_peaks": 3,
    "build_min_rise_db": 3.0,
    "build_lookback_seconds": 24.0,
    "fall_min_db": 2.5,
    "drop_min_db": 4.0,
    "drop_max_seconds": 6.0,
    "release_min_seconds": 10.0,
    "fall_lookahead_seconds": 18.0,
}

BANDS = {"low": (20.0, 250.0), "mid": (200.0, 2000.0), "high": (2000.0, 8000.0)}


def _cfg_value(cfg: dict | None, key: str):
    if cfg:
        v = cfg.get("analysis", {}).get(key)
        if v is not None:
            return v
    return DEFAULTS[key]


def frame_curves(y: np.ndarray, sr: int, cfg: dict | None = None) -> dict:
    """One STFT, several frame-level curves. Shared with `structure` so nothing computes twice.

    Returns arrays all aligned to `n_frames`, plus `times` in seconds.
    """
    import librosa

    fl = int(_cfg_value(cfg, "frame_length"))
    hop = int(_cfg_value(cfg, "hop_length"))
    lo_hz, hi_hz = _cfg_value(cfg, "low_band_hz")

    stft = librosa.stft(y, n_fft=fl, hop_length=hop)
    mag = np.abs(stft)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=fl)

    n = power.shape[1]
    total = power.sum(axis=0) + 1e-12
    # Time-domain frame RMS, same call the rest of the codebase uses. Deriving RMS from
    # `power.mean()` looks equivalent but is not: librosa's STFT carries its own scale, so the
    # numbers were neither dBFS nor comparable to the stored rms_mean.
    rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hop)[0]
    rms = _align(rms, n)
    rms_db = np.maximum(20.0 * np.log10(rms + 1e-9), float(_cfg_value(cfg, "rms_db_floor")))

    def band_share(lo: float, hi: float) -> np.ndarray:
        m = (freqs >= lo) & (freqs < hi)
        return power[m].sum(axis=0) / total if m.any() else np.zeros(n)

    bright = (freqs[:, None] * power).sum(axis=0) / total
    # 旧包络：线性幅度谱直接当 `S`。librosa 文档里 onset_strength 的 `S` 应是
    # 对数功率梅尔谱，所以这份数值"看着能用"但与文档约定不一致——结构分组的
    # 相关阈值（chorus_min_onset_pct 等）是按它标定的，保留给结构兼容，字段
    # 上只作为 activity_proxy 对外解释。
    onset = librosa.onset.onset_strength(S=mag, sr=sr, hop_length=hop, backtrack=False)
    onset = _align(onset, n)
    # 按 librosa 约定直接从音频算的包络与事件列表：onset_rate_hz 与维度事件
    # 用这一份，保证"次/秒"有明确含义且跨模块一致。
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_env = _align(onset_env, n)
    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop, backtrack=False, units="time")
    # spectral flux: positive spectral flux per frame, divided by that frame's own
    # magnitude mass, so a loud frame is not automatically a "changing" frame.
    diff = np.maximum(0.0, np.diff(mag, axis=1, prepend=mag[:, :1]))
    change = _align(diff.sum(axis=0) / (mag.sum(axis=0) + 1e-9), n)

    chroma = librosa.feature.chroma_stft(S=power, n_fft=fl, hop_length=hop, sr=sr, norm=2)

    return {
        "times": librosa.times_like(rms, sr=sr, hop_length=hop),
        "hop_length": hop,
        "frame_length": fl,
        "n_frames": int(n),
        "rms": rms,
        "rms_db": rms_db,
        "onset": onset,
        "onset_env": onset_env,
        "onset_times": [round(float(t), 3) for t in np.asarray(onset_times, float)],
        "low": band_share(lo_hz, hi_hz),
        "bright": bright,
        "change": change,
        "chroma": chroma,
        "bands": {k: band_share(*v) for k, v in BANDS.items()},
    }


def _align(a: np.ndarray, n: int) -> np.ndarray:
    """Trim/pad a frame series to exactly n frames (onset/flux can be one off)."""
    a = np.asarray(a, float).ravel()
    if a.size == n:
        return a
    if a.size > n:
        return a[:n]
    return np.pad(a, (0, n - a.size), constant_values=0.0)


def grid(times: np.ndarray, curves: dict, interval: float) -> dict:
    """Re-sample frame series onto a fixed second grid (mean per bucket).

    Fixed-interval buckets with the interval stored in metadata: an evenly averaged
    64-value blob cannot be turned back into seconds, which is why the old
    `energy_curve` could not be shown on a timeline.
    """
    dur = float(times[-1]) if len(times) else 0.0
    # exactly the buckets the track covers: rounding up by one extra bucket used to put the
    # last point past the end of the song, which a timeline then draws as phantom silence.
    n_buckets = max(1, int(np.ceil(dur / interval))) if dur > 0 else 1
    out = {"time_seconds": [round((i + 0.5) * interval, 3) for i in range(n_buckets)]}
    idx = np.clip((np.asarray(times, float) / interval).astype(int), 0, n_buckets - 1)
    counts = np.bincount(idx, minlength=n_buckets).astype(float)
    empty = counts == 0
    for key in ("rms_db", "onset", "low", "bright", "change"):
        vals = np.bincount(idx, weights=curves[key], minlength=n_buckets)
        vals = np.divide(vals, counts, out=np.zeros_like(vals), where=counts > 0)
        vals[empty] = np.nan
        out[key] = _fill_and_round(vals, key)
    # onset_rate_hz: real detected onset events per bucket, divided by the bucket's
    # true duration. Zero is a genuine measurement (no onsets in that bucket), so this
    # series is never NaN-filled or interpolated the way the frame-mean curves are.
    ev = np.asarray(curves.get("onset_times") or [], float)
    ev_idx = np.clip((ev / interval).astype(int), 0, n_buckets - 1) if ev.size else np.array([], int)
    rate = np.bincount(ev_idx, minlength=n_buckets).astype(float) / interval
    out["onset_rate"] = [round(float(x), 3) for x in rate]
    out["rms_norm"] = _minmax(out["rms_db"])
    counts_out = np.where(empty, 0.0, counts)
    out["frames_per_bucket"] = [int(c) for c in counts_out]
    return out


def _fill_and_round(vals: np.ndarray, key: str) -> list[float]:
    v = np.asarray(vals, float).copy()
    if np.isnan(v).all():
        return [0.0] * v.size
    # a bucket with no frames (silence-long gaps are rare but possible) takes the
    # nearest real value instead of inventing 0, which would read as a dropout.
    good = ~np.isnan(v)
    v[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), v[good])
    nd = 2 if key == "rms_db" else 4
    return [round(float(x), nd) for x in v]


def _minmax(vals: list[float]) -> list[float]:
    v = np.asarray(vals, float)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return [0.5] * len(vals)
    return [round(float((x - lo) / (hi - lo)), 4) for x in v]


def _moving_average(v: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return v
    pad = width // 2
    padded = np.pad(v, (pad, pad), mode="edge")
    kernel = np.ones(width, float) / width
    return np.convolve(padded, kernel, mode="valid")[: v.size]


def events(grid_curves: dict, cfg: dict | None = None) -> list[dict]:
    """Locate build / peak / pull-back / drop / release on the smoothed loudness curve.

    A peak must be corroborated by at least one other feature (onset activity, low band,
    spectral brightness, arrangement change) to earn more than the floor confidence:
    a single loud frame is not a climax. Thresholds are relative to the track's own
    dynamic range so a mastered EDM track and a solo guitar recording are both readable.
    """
    from scipy.signal import find_peaks, peak_prominences

    interval = float(_cfg_value(cfg, "curve_interval_seconds"))
    times = np.asarray(grid_curves.get("time_seconds") or [], float)
    level_db = np.asarray(grid_curves.get("rms_db") or [], float)
    if times.size < 8 or level_db.size != times.size:
        return []

    width = max(1, int(round(_cfg_value(cfg, "smooth_seconds") / interval)))
    level = _moving_average(level_db, width)
    rng = float(np.percentile(level, 95) - np.percentile(level, 5))
    if rng < float(_cfg_value(cfg, "min_dynamic_range_db")):
        # essentially flat recording: there is no build/drop to report, and inventing one
        # from 0.2 dB of metering noise is exactly the failure this module exists to stop.
        return []

    corr = {
        "onset_activity": np.asarray(grid_curves.get("onset") or [], float),
        "low_band_energy": np.asarray(grid_curves.get("low") or [], float),
        "spectral_brightness": np.asarray(grid_curves.get("bright") or [], float),
        "arrangement_change": np.asarray(grid_curves.get("change") or [], float),
    }

    prominence_floor = max(1.0, float(_cfg_value(cfg, "peak_prominence_fraction")) * rng)
    dist = max(1, int(round(_cfg_value(cfg, "peak_min_distance_seconds") / interval)))
    pk, _ = find_peaks(level, prominence=prominence_floor, distance=dist)
    if pk.size == 0:
        pk = np.array([int(np.argmax(level))])
    proms = np.asarray(peak_prominences(level, pk)[0], float) if pk.size else np.array([])
    keep = pk[np.argsort(proms)][-_cfg_value(cfg, "max_peaks"):] if pk.size else pk
    out: list[dict] = []

    look_back = max(1, int(round(_cfg_value(cfg, "build_lookback_seconds") / interval)))
    look_ahead = max(1, int(round(_cfg_value(cfg, "fall_lookahead_seconds") / interval)))
    rise_min = float(_cfg_value(cfg, "build_min_rise_db"))
    fall_min = float(_cfg_value(cfg, "fall_min_db"))
    drop_db = float(_cfg_value(cfg, "drop_min_db"))
    drop_fast = float(_cfg_value(cfg, "drop_max_seconds"))
    release_slow = float(_cfg_value(cfg, "release_min_seconds"))

    for p in sorted(int(x) for x in keep):
        supports = [name for name, arr in corr.items()
                    if arr.size == times.size and _rose_into(arr, p, look_back)]
        peak_conf = _peak_confidence(times, level, p, rng, look_back, len(supports))
        out.append({
            "time": round(float(times[p]), 2),
            "type": "peak",
            "confidence": peak_conf,
            "evidence": [f"rms_peak:{level[p]:.1f}dB", f"dynamic_range:{rng:.1f}dB"]
                        + [f"{_snake(s)}_also_rises" for s in supports],
        })

        lo = max(0, p - look_back)
        if p - lo >= 2:
            window = level[lo:p + 1]
            t0 = int(np.argmin(window))
            rise = float(window[-1] - window[t0])
            if rise >= rise_min:
                out.append({
                    "time": round(float(times[lo + t0]), 2),
                    "type": "build",
                    "confidence": round(min(0.9, 0.3 + 0.6 * rise / (2 * rng)), 3),
                    "evidence": [f"rms_rise:+{rise:.1f}dB_over_{(p - lo - t0) * interval:.0f}s"]
                                + (["density_increases"]
                                   if _rose_into(corr["onset_activity"], p, look_back) else []),
                })

        hi = min(times.size, p + 1 + look_ahead)
        if hi - (p + 1) >= 2:
            window = level[p:hi]
            t1 = int(np.argmin(window))
            fall = float(window[0] - window[t1])
            span = float(times[p + t1] - times[p]) if t1 < window.size else 0.0
            if fall >= fall_min:
                if fall >= drop_db and span <= drop_fast:
                    etype = "drop"
                elif span >= release_slow:
                    etype = "release"
                else:
                    etype = "pull-back"
                out.append({
                    "time": round(float(times[p + t1]), 2),
                    "type": etype,
                    "confidence": round(min(0.9, 0.3 + 0.6 * fall / (2 * rng)), 3),
                    "evidence": [f"rms_fall:-{fall:.1f}dB_over_{span:.0f}s", f"from_peak:{round(float(times[p]), 1)}s"],
                })

    out.sort(key=lambda e: (e["time"], e["type"]))
    return out


def _snake(name: str) -> str:
    return name.replace("-", "_").lower()


def _rose_into(arr: np.ndarray, p: int, look_back: int) -> bool:
    lo = max(0, p - look_back)
    seg = arr[lo:p + 1]
    if seg.size < 3 or not np.any(seg):
        return False
    base = float(np.percentile(arr, 25))
    spread = float(np.percentile(arr, 75) - base)
    if spread <= 1e-9:
        return False
    return float(seg[-1]) >= base + 0.75 * spread


def _peak_confidence(times, level, p, rng, look_back, n_supports) -> float:
    lo = max(0, p - look_back)
    rise = float(level[p] - level[lo:p + 1].min())
    # relative height inside the track, plus how many independent features agree.
    score = 0.30 + 0.45 * min(1.0, rise / max(rng, 1e-6)) + 0.075 * n_supports
    return round(float(min(0.9, max(0.2, score))), 3)


# 每个可检测维度：曲线键 -> (单位, 最小可报告变化幅度)。幅度阈值让"这一维度
# 本来就平"的轴安静地跳过，而不是把计量噪声放大成事件；一条轴平了不连累
# 其他轴——这正是电平平稳的母带歌也会换音色 / 换律动的原因。
_AXIS_SPECS: dict = {
    "onset_rate_hz": ("events/s", 0.20),
    "onset_activity": ("activity_proxy", 0.02),
    "low_band_ratio": ("ratio 0..1", 0.02),
    "spectral_brightness": ("Hz", 0.0),       # 用中位数的 2% 兜底
    "spectral_flux": ("normalized 0..1", 0.01),
}


def detect_axis_changes(g: dict, cfg: dict | None = None) -> list[dict]:
    """Per-axis change events, independent of the loudness curve.

    `events()` above only looks at level; a heavily compressed master with a
    nearly flat dB curve still swaps texture / groove / brightness, and those
    changes used to be invisible. Each axis is smoothed and checked for
    (a) sudden drops inside `drop_max_seconds` and (b) one sustained rise or
    fall across the track. Missing / short series are skipped per axis.
    """
    interval = float(_cfg_value(cfg, "curve_interval_seconds"))
    times = np.asarray(g.get("time_seconds") or [], float)
    n = int(times.size)
    if n < 8:
        return []

    width = max(1, int(round(_cfg_value(cfg, "smooth_seconds") / interval)))
    short_w = max(1, int(round(2.0 / interval)))
    drop_window = max(1, int(round(_cfg_value(cfg, "drop_max_seconds") / interval)))
    edge = max(1, int(round(2.0 / interval)))  # 首尾各 2 秒：起播瞬态 / 结尾截断不算乐思变化
    out: list[dict] = []

    for axis, (unit, min_range) in _AXIS_SPECS.items():
        raw = np.asarray(g.get(_grid_key(axis)) or [], float)
        if raw.size != n:
            continue
        # 起音类维度：活动几乎只出现在起播瞬间的轴没有可报告的"变化"，
        # 跳过，别把检测器在静态内容上的零星毛刺当成事件。
        if axis in ("onset_rate_hz", "onset_activity"):
            if float(np.count_nonzero(raw)) / n < 0.05:
                continue
        v = _moving_average(raw, width)
        spread = float(np.percentile(v, 95) - np.percentile(v, 5))
        if axis == "spectral_brightness":
            floor = max(min_range, 0.02 * float(np.median(v[np.isfinite(v)]) or 0.0))
        else:
            floor = min_range
        rng = max(spread, floor)
        if spread < floor:
            continue  # this axis is flat: skip it, keep the other axes

        # --- sudden drops: fall of the short-smoothed series within drop_max_seconds
        s = _moving_average(raw, short_w)
        step = max(1, drop_window // 2)
        diffs = s[step:] - s[:n - step] if n > step else np.array([])
        taken: list[int] = []
        order = np.argsort(diffs) if diffs.size else np.array([], int)
        for k in order:
            if len(taken) >= 3:
                break
            if float(diffs[k]) > -max(0.5 * rng, floor):
                break
            i = int(k)
            if i < edge or i + step > n - edge:
                continue  # 首尾 2 秒是瞬态/截断的领地，不是乐思变化
            if any(abs(i - t) < step for t in taken):
                continue
            before, after = float(s[i]), float(s[i + step])
            # 起音率的"骤降"必须从真实活动掉下来：事件前至少 1 次/秒，且掉到
            # 此前的一半以下——否则恒定律动背景上的 ±1 次抖动会被当成事件。
            if axis == "onset_rate_hz" and (before < 1.0 or after > 0.5 * before):
                continue
            taken.append(i)
            out.append({
                "time": round(float(times[min(n - 1, i + step // 2)]), 2),
                "type": "axis-drop",
                "axis": axis,
                "unit": unit,
                "confidence": round(min(0.9, 0.3 + 0.5 * abs(float(diffs[k])) / max(rng, 1e-9)), 3),
                "evidence": [f"axis:{axis}", f"change:{before:.3g}->{after:.3g} {unit}",
                             f"fall_within:{_cfg_value(cfg, 'drop_max_seconds'):g}s"],
            })

        # --- one sustained drift across the track (rise or fall of the whole axis)
        quarter = max(2, n // 4)
        if n >= 4 * quarter // 2 and quarter * 2 <= n:
            head = float(np.mean(v[:quarter]))
            tail = float(np.mean(v[n - quarter:]))
            delta = tail - head
            if abs(delta) >= max(0.6 * rng, floor):
                out.append({
                    "time": round(float(times[n // 2]), 2),
                    "type": "axis-rise" if delta > 0 else "axis-fall",
                    "axis": axis,
                    "unit": unit,
                    "confidence": round(min(0.9, 0.3 + 0.5 * abs(delta) / max(rng, 1e-9)), 3),
                    "evidence": [f"axis:{axis}",
                                 f"first_quarter:{head:.3g}{unit}",
                                 f"last_quarter:{tail:.3g}{unit}",
                                 f"drift:{delta:+.3g}{unit}"],
                })

    # 顺序保持时间序；单轴数量已各自封顶，这里不再次裁剪。
    out.sort(key=lambda e: (e["time"], e["type"]))
    return out


def _grid_key(axis: str) -> str:
    """曲线输出键：onset_rate_hz 存在 grid 里叫 onset_rate，其余同名。"""
    return {"onset_rate_hz": "onset_rate"}.get(axis, {
        "onset_activity": "onset",
        "low_band_ratio": "low",
        "spectral_brightness": "bright",
        "spectral_flux": "change",
    }.get(axis, axis))


def track_dynamics(y: np.ndarray, sr: int, cfg: dict | None = None,
                   frames: dict | None = None) -> dict:
    """Analyse the complete track and return the `dynamics` block of the analysis JSON."""
    interval = float(_cfg_value(cfg, "curve_interval_seconds"))
    frames = frames if frames is not None else frame_curves(y, sr, cfg)
    if frames["n_frames"] == 0:
        return {}
    duration = float(len(y) / sr)
    g = grid(frames["times"], frames, interval)
    # 电平事件照旧只看响度；维度事件独立检测，电平平稳不再屏蔽
    # 音色 / 律动 / 亮度上的变化。
    ev = events(g, cfg) + detect_axis_changes(g, cfg)
    ev.sort(key=lambda e: (e["time"], e["type"]))
    level_db = np.asarray(g["rms_db"], float)
    return {
        "time_seconds": g["time_seconds"],
        "rms_db": g["rms_db"],
        "rms_norm": g["rms_norm"],
        # deprecated 命名，保留只为兼容旧客户端：这是 onset 包络的桶内积分
        # 代理量，不是实际起音次数；次数请用 onset_rate_hz。
        "onset_density": [round(float(x) * interval, 4) for x in g["onset"]],
        "onset_rate_hz": g["onset_rate"],
        # 新的规范名是 low_band_ratio（低频功率占比）；low_band_energy 是旧别名。
        "low_band_ratio": g["low"],
        "low_band_energy": g["low"],
        "spectral_brightness": [round(float(x), 2) for x in g["bright"]],
        # 规范名 spectral_flux（谱流，算法名）；arrangement_change 是旧别名，
        # 它从来测不了"编曲层数"。
        "spectral_flux": g["change"],
        "arrangement_change": g["change"],
        "events": ev,
        "meta": {
            "scope": "full_track",
            "duration_seconds": round(duration, 3),
            "interval_seconds": interval,
            "n_points": len(g["time_seconds"]),
            # what a client must know to plot the curve on a timeline without guessing
            "time_semantics": "bucket_center",
            "rms_db_unit": "dBFS",
            "rms_db_floor": float(_cfg_value(cfg, "rms_db_floor")),
            "frame_length": frames["frame_length"],
            "hop_length": frames["hop_length"],
            "level_mean_db": round(float(level_db.mean()), 2),
            "level_range_db": round(float(np.percentile(level_db, 95)
                                          - np.percentile(level_db, 5)), 2),
            "confidence_kind": "model_score",
            "calibrated": False,
            "notes": [
                "onset_density 已弃用：是旧包络（线性幅度谱约定，保留给结构分组标定兼容）的积分代理量，不是起音次数；次数看 onset_rate_hz（次/秒）",
                "onset_rate_hz 是实际检测到的起音事件数 / 桶真实时长，事件按 librosa 约定直接从音频检测",
                "low_band_ratio 是 20-250 Hz 能量占比（low_band_energy 是旧别名），不等于 bass 乐器",
                "spectral_brightness 是功率加权频率质心，单位 Hz",
                "spectral_flux 是归一化谱流（arrangement_change 是旧别名），只说明音色在变，不说明是哪种变化",
                "axis-rise / axis-fall / axis-drop 事件来自各维度曲线的独立检测，与电平无关；电平平稳的歌照样可以有维度变化",
                "confidence 是未校准模型分数，人工标注前不可当概率读",
            ],
        },
    }
