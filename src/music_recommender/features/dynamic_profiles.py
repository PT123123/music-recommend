"""Composable whole-track dynamic profiles: 12 multi-label attributes, NOT genres.

Every profile is attached to a metric axis (level, onset rate, low-band ratio,
brightness, flux), a time interval, a score and evidence strings. They are
multi-label and combinable: "A/B 交替对比（电平）+ B 组第三次出现低频增强 +
尾段电平渐弱" is a meaningful combination; "先抑后扬" alone is not.

Honesty rules (expansion doc section 5):
  - the 12 names are a communication vocabulary, not a recognised taxonomy and
    not claimed to cover any percentage of real playlists;
  - an axis can be flat while another axis moves; profiles never claim "the
    whole song is flat" from the level axis alone;
  - ending-stop must exclude trailing silence, file truncation and padding; if
    that cannot be distinguished, the score is lowered and the reason recorded.
"""
from __future__ import annotations

import numpy as np

# 轮廓检测用的轴：曲线键 -> 显示名
AXES: dict[str, str] = {
    "level_db": "电平",
    "onset_rate_hz": "起音率",
    "low_band_ratio": "低频比例",
    "spectral_brightness": "亮度",
    "spectral_flux": "谱流",
}

# 每个轴上判定"平"的阈值（与 dynamics.detect_axis_changes 的 floor 同思想）
_AXIS_FLOOR: dict[str, float] = {
    "level_db": 1.5,
    "onset_rate_hz": 0.2,
    "low_band_ratio": 0.02,
    "spectral_brightness": 0.0,   # 用中位数 2% 兜底
    "spectral_flux": 0.01,
}


def _series(curves: dict, axis: str) -> np.ndarray | None:
    key = {"level_db": "rms_db", "onset_rate_hz": "onset_rate_hz",
           "low_band_ratio": "low_band_ratio", "spectral_brightness": "spectral_brightness",
           "spectral_flux": "spectral_flux"}.get(axis)
    vals = curves.get(key)
    if not isinstance(vals, list) or len(vals) < 8:
        return None
    v = np.asarray(vals, float)
    return v if np.isfinite(v).all() else None


def summarize_dynamic_profiles(curves: dict, segments: list[dict],
                               events: list[dict], cfg: dict | None = None) -> dict:
    times = np.asarray(curves.get("time_seconds") or [], float)
    duration = float(times[-1]) if times.size else 0.0
    out: list[dict] = []
    if times.size < 8 or duration <= 0:
        return {"profiles": [], "meta": {"scope": "full_track", "note": "曲线太短，轮廓不判定"}}

    for axis, label in AXES.items():
        v = _series(curves, axis)
        if v is None:
            continue
        spread = float(np.percentile(v, 95) - np.percentile(v, 5))
        floor = _AXIS_FLOOR.get(axis, 0.0)
        if axis == "spectral_brightness":
            floor = max(floor, 0.02 * float(np.median(v)))
        rng = max(spread, floor)
        if spread < floor:
            # 这条轴全曲几乎恒定：给一条明确的"平稳"记录，而不是沉默——
            # "电平平稳"不能说成"整首歌所有维度都平稳"，所以逐轴陈述。
            out.append(_profile("全曲平稳", axis, label, 0.0, times,
                                evidence=[f"range:{spread:.3g}", "below_report_floor"]))
            continue
        n = v.size
        seg_means = _segment_means(times, v, segments)

        # --- 逐段增加 / 逐段减少（至少 3 段且单调性明显）
        if len(seg_means) >= 3:
            deltas = np.diff([m for _s, _e, m in seg_means])
            up = float(np.mean(deltas > 0))
            down = float(np.mean(deltas < 0))
            if up >= 0.8 and deltas.mean() > 0.05 * rng:
                out.append(_profile("逐段增加", axis, label, min(0.9, 0.4 + 0.4 * up),
                                    times, interval=[seg_means[0][0], seg_means[-1][1]],
                                    evidence=[f"rising_segments:{sum(deltas > 0)}/{len(deltas)}",
                                              f"total_change:{deltas.sum():.3g}"]))
            elif down >= 0.8 and deltas.mean() < -0.05 * rng:
                out.append(_profile("逐段减少", axis, label, min(0.9, 0.4 + 0.4 * down),
                                    times, interval=[seg_means[0][0], seg_means[-1][1]],
                                    evidence=[f"falling_segments:{sum(deltas < 0)}/{len(deltas)}",
                                              f"total_change:{deltas.sum():.3g}"]))

        # --- 交替对比：相邻段高低交替且落差明显
        if len(seg_means) >= 4:
            hi_lo = [m > np.mean([x for _s, _e, x in seg_means]) for _s, _e, m in seg_means]
            flips = sum(1 for a, b in zip(hi_lo, hi_lo[1:]) if a != b)
            if flips >= len(seg_means) - 2:
                name = "A/B 交替对比"
                out.append(_profile(name, axis, label, min(0.9, 0.4 + 0.1 * flips),
                                    times,
                                    evidence=[f"alternations:{flips}/{len(seg_means) - 1}",
                                              "groups_named_A/B_only（未确认主副歌）"]))

        # --- 开场即强：开头一段有效音乐已接近全曲高位
        if seg_means:
            s0, e0, m0 = seg_means[0]
            if m0 >= np.percentile(v, 80) and s0 <= duration * 0.3:
                out.append(_profile("开场即强", axis, label,
                                    min(0.9, 0.3 + 0.5 * (m0 - np.percentile(v, 50)) / max(rng, 1e-9)),
                                    times, interval=[s0, seg_means[0][1]],
                                    evidence=[f"opening_mean:{m0:.3g}",
                                              f"track_p80:{np.percentile(v, 80):.3g}"]))

        # --- 单次 / 多次突出高点：显著高位区间
        hi_intervals = _high_intervals(times, v, rng)
        hi_intervals = [iv for iv in hi_intervals if iv[2] >= max(4.0, duration * 0.04)]
        if len(hi_intervals) == 1:
            out.append(_profile("单次突出高点", axis, label, min(0.9, 0.4 + 0.4 * (hi_intervals[0][3] / max(rng, 1e-9))),
                                times, interval=[hi_intervals[0][0], hi_intervals[0][1]],
                                evidence=[f"peak_position:{(hi_intervals[0][0] + hi_intervals[0][1]) / 2:.0f}s",
                                          f"height_above_median:{hi_intervals[0][3]:.3g}"]))
        elif len(hi_intervals) >= 2:
            out.append(_profile("多次突出高点", axis, label, min(0.9, 0.35 + 0.15 * len(hi_intervals)),
                                times,
                                evidence=[f"n_high_regions:{len(hi_intervals)}",
                                          f"positions:{','.join(f'{(a + b) / 2:.0f}s' for a, b, _, _ in hi_intervals[:4])}"]))

        # --- 高点后回落：主高点之后持续下降
        if hi_intervals:
            a, b, _, _ = hi_intervals[-1]
            after = v[(times >= b) & (times <= duration)]
            if after.size >= 4:
                fall = float(after[: max(2, after.size // 2)].mean() - after[-max(2, after.size // 2):].mean())
                if fall >= max(0.3 * rng, floor):
                    out.append(_profile("高点后回落", axis, label,
                                        min(0.9, 0.35 + 0.4 * fall / max(rng, 1e-9)),
                                        times, interval=[b, duration],
                                        evidence=[f"fall_after_peak:{fall:.3g}"]))

        # --- 抽空再恢复：中段下降后回到此前水平
        half = n // 2
        if half >= 4:
            lo_window = v[half // 2: half + half // 2]
            if lo_window.size >= 2:
                dip = float(np.mean(v[:half // 2]) - lo_window.min())
                recover = float(lo_window.min() - np.mean(v[-half // 2:])) * -1
                if dip >= max(0.4 * rng, floor) and abs(float(lo_window.min() - np.mean(v[-half // 2:]))) <= 0.2 * rng:
                    dip_t = float(times[half // 2 + int(np.argmin(lo_window))])
                    out.append(_profile("抽空再恢复", axis, label,
                                        min(0.9, 0.35 + 0.4 * dip / max(rng, 1e-9)),
                                        times, interval=[dip_t - 4.0, dip_t + 4.0],
                                        evidence=[f"dip:{dip:.3g}", f"at:{dip_t:.0f}s",
                                                  "recovered_to_prior_level"]))

    # --- 与轴无关的轮廓需要段落分组信息
    if segments:
        out += _group_profiles(curves, segments, times)

    # --- 结尾轮廓：骤停 / 渐弱 / 延展（用起音率 + 电平联合判断）
    out += _ending_profiles(curves, times, duration)

    return {
        "profiles": out,
        "meta": {
            "scope": "full_track",
            "note": "轮廓是多标签属性组合，不是互斥曲风类别；每条都带维度、区间、分数与证据",
        },
    }


def _profile(name: str, axis: str, axis_label: str, score: float, times: np.ndarray,
             interval: list[float] | None = None, evidence: list | None = None) -> dict:
    t0 = round(float(times[0]), 1)
    t1 = round(float(times[-1]), 1)
    return {
        "profile": name,
        "axis": axis,
        "axis_label": axis_label,
        "score": round(float(min(0.95, max(0.05, score))), 3),
        "interval": interval or [t0, t1],
        "evidence": evidence or [],
    }


def _segment_means(times: np.ndarray, v: np.ndarray, segments: list[dict]) -> list[tuple[float, float, float]]:
    rows = []
    for seg in segments:
        s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if e <= s:
            continue
        m = (times >= s) & (times < e)
        if m.sum() < 2:
            continue
        rows.append((s, e, float(v[m].mean())))
    return rows


def _high_intervals(times: np.ndarray, v: np.ndarray, rng: float) -> list[tuple[float, float, float, float]]:
    """显著高位区间：连续高于中位数 + 0.5*rng 的段，宽度并入相邻段。"""
    med = float(np.median(v))
    thr = med + 0.5 * rng
    mask = v > thr
    ivs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ivs.append((start, i))
            start = None
    if start is not None:
        ivs.append((start, len(mask)))
    out = []
    for a, b in ivs:
        if b - a < 2:
            continue
        out.append((float(times[a]), float(times[b - 1]), b - a, float(v[a:b].mean() - med)))
    return out


def _group_profiles(curves: dict, segments: list[dict], times: np.ndarray) -> list[dict]:
    """重复段落加码：同一组后几次出现的指定维度更高。"""
    groups: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        g = seg.get("group", "")
        if not g:
            continue
        groups.setdefault(g, []).append((float(seg.get("start", 0.0)), float(seg.get("end", 0.0))))
    out = []
    for g, spans in groups.items():
        if len(spans) < 2:
            continue
        for axis, key in (("low_band_ratio", "low_band_ratio"), ("onset_rate_hz", "onset_rate_hz"),
                          ("spectral_brightness", "spectral_brightness")):
            v = _series(curves, axis)
            if v is None:
                continue
            means = []
            for s, e in spans:
                m = (times >= s) & (times < e)
                if m.sum() < 2:
                    means = []
                    break
                means.append(float(v[m].mean()))
            if not means:
                continue
            if means[-1] > means[0] * 1.15 + 1e-9:
                out.append({
                    "profile": "重复段落加码",
                    "axis": axis,
                    "axis_label": AXES.get(axis, axis),
                    "score": round(min(0.9, 0.35 + 0.2 * (means[-1] / max(means[0], 1e-9))), 3),
                    "interval": [spans[-1][0], spans[-1][1]],
                    "evidence": [f"group:{g}", f"first:{means[0]:.3g}", f"last:{means[-1]:.3g}",
                                 f"occurrence:{len(means)}"],
                })
                break  # 一组报一条最明显的维度就够了
    return out


def _ending_profiles(curves: dict, times: np.ndarray, duration: float) -> list[dict]:
    """结尾骤停 / 尾段延展 / 渐弱。"""
    out = []
    rate = np.asarray(curves.get("onset_rate_hz") or [], float)
    level = np.asarray(curves.get("rms_db") or [], float)
    if times.size < 8 or rate.size != times.size or level.size != times.size:
        return out
    tail_s = max(4.0, duration * 0.08)
    body = times <= duration - tail_s
    if body.sum() < 4:
        return out
    body_rate = float(np.mean(rate[body]))
    tail_rate = float(np.mean(rate[~body]))
    tail_level = float(np.mean(level[~body]))
    body_level = float(np.mean(level[body]))
    # 真正的"骤停"要求结尾几乎无起音且电平也明显低；只满足其中一条时降低分数
    silent_rate = tail_rate < max(0.25 * body_rate, 0.1)
    quiet_level = tail_level < body_level - 6.0
    if silent_rate and quiet_level:
        out.append({"profile": "结尾骤停", "axis": "onset_rate_hz+level_db", "axis_label": "起音率+电平",
                    "score": 0.75, "interval": [round(duration - tail_s, 1), round(duration, 1)],
                    "evidence": [f"tail_onset_rate:{tail_rate:.2f}/s vs body:{body_rate:.2f}/s",
                                 f"tail_level:{tail_level:.1f}dB vs body:{body_level:.1f}dB"]})
    elif silent_rate or quiet_level:
        out.append({"profile": "结尾骤停(弱证据)", "axis": "onset_rate_hz+level_db", "axis_label": "起音率+电平",
                    "score": 0.35, "interval": [round(duration - tail_s, 1), round(duration, 1)],
                    "evidence": [f"tail_onset_rate:{tail_rate:.2f}/s vs body:{body_rate:.2f}/s",
                                 f"tail_level:{tail_level:.1f}dB vs body:{body_level:.1f}dB",
                                 "single_axis_only: 无法排除文件截断或补零，分数已降"]})
    # 渐弱：尾段电平相对中后段缓慢下降（≥ 3 dB 且不是骤停）
    mid = (times >= duration * 0.5) & (times <= duration - tail_s)
    if mid.sum() >= 4 and not (silent_rate and quiet_level):
        drop = float(np.mean(level[mid]) - tail_level)
        if drop >= 3.0:
            out.append({"profile": "尾段渐弱", "axis": "level_db", "axis_label": "电平",
                        "score": min(0.9, 0.3 + 0.1 * drop), "interval": [round(duration - tail_s, 1), round(duration, 1)],
                        "evidence": [f"level_drop:{drop:.1f}dB_over_tail"]})
    return out
