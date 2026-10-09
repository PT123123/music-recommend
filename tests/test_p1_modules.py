"""P1 扩展模块的合成信号验收（expansion doc section 14 的可检样本）。

恒定电平但音色变亮、单声道不伪造空间分析、双单声道 vs 真立体声、恒定 BPM
但起音变密不加速、轮廓逐段上升、描述不发明"副歌"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from music_recommender.features import description, spatial, texture  # noqa: E402
from music_recommender.features import dynamic_profiles as dprof  # noqa: E402
from music_recommender.features import dynamics, rhythm  # noqa: E402

SR = 22050


def _tone(freq: float, seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _click_track(bpm: float, seconds: float) -> np.ndarray:
    y = np.zeros(int(seconds * SR))
    period = int(60.0 / bpm * SR)
    click_len = int(0.01 * SR)
    for start in range(0, len(y) - click_len, period):
        y[start:start + click_len] = 0.8 * np.exp(-np.linspace(0, 6, click_len))
    return y


# ---------------------------------------------------------------- spatial

def test_dual_mono_vs_real_stereo():
    tone = _tone(440.0, 20.0)
    dual = np.stack([tone, tone])
    out = spatial.spatial_curves(dual, SR, {"analysis": {}})
    assert out["status"] == "ready"
    corr = [v for v in out["curves"]["lr_correlation"] if v is not None]
    side = [v for v in out["curves"]["side_energy_ratio"] if v is not None]
    assert min(corr) > 0.99, "双单声道必须读出接近 +1 的相关度"
    assert max(side) < 0.05, "双单声道的 side 占比应近零——这是测量，不是缺数据"

    # 左右不同内容：相关度下降、side 占比上升
    other = _tone(660.0, 20.0)
    stereo = np.stack([tone, other])
    out2 = spatial.spatial_curves(stereo, SR, {"analysis": {}})
    corr2 = [v for v in out2["curves"]["lr_correlation"] if v is not None]
    side2 = [v for v in out2["curves"]["side_energy_ratio"] if v is not None]
    assert np.mean(corr2) < 0.5
    assert np.mean(side2) > 0.3


def test_mono_reports_unavailable_and_multichannel_refused():
    out = spatial.spatial_curves(_tone(440.0, 10.0), SR, {"analysis": {}})
    assert out["status"] == "unavailable" and out["reason"] == "mono"
    three = np.stack([_tone(440.0, 10.0), _tone(550.0, 10.0), _tone(660.0, 10.0)])
    out2 = spatial.spatial_curves(three, SR, {"analysis": {}})
    assert out2["status"] == "unavailable"
    assert "multichannel" in out2["reason"], "多声道必须拒绝，不得擅自取前两声道"


def test_silent_buckets_are_invalid_not_zero():
    y = np.concatenate([_tone(440.0, 8.0, 0.3), np.zeros(int(8 * SR))])
    out = spatial.spatial_curves(np.stack([y, y]), SR, {"analysis": {}})
    assert out["status"] == "ready"
    valid = out["valid"]
    assert any(valid), "有声段必须有效"
    assert not all(valid), "静音段必须无效"
    for i, ok in enumerate(valid):
        if not ok:
            for curve in out["curves"].values():
                assert curve[i] is None, "无效桶必须是 None，不能伪装成完美居中"


# ---------------------------------------------------------------- texture

def test_noise_is_flatter_than_sine():
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(SR * 10) * 0.1
    sine = _tone(440.0, 10.0)
    flat_noise = texture.texture_curves(noise, SR, {"analysis": {}})["spectral_flatness"]
    flat_sine = texture.texture_curves(sine, SR, {"analysis": {}})["spectral_flatness"]
    assert np.mean(flat_noise) > np.mean(flat_sine) * 5, "噪声的平坦度应远高于纯正弦"


def test_texture_brighter_signal_higher_hf_ratio():
    low = texture.texture_curves(_tone(220.0, 8.0), SR, {"analysis": {}})["hf_power_ratio"]
    bright = texture.texture_curves(_tone(6000.0, 8.0), SR, {"analysis": {}})["hf_power_ratio"]
    assert np.mean(bright) > np.mean(low), "6kHz 正弦的高频占比应远高于 220Hz"


# ---------------------------------------------------------------- rhythm

def test_constant_click_stays_constant_even_when_denser():
    steady = rhythm.rhythm_timeline(_click_track(120.0, 40.0), SR, {"analysis": {}})
    assert steady["status"] == "ready"
    assert steady["tempo_stable"], "恒定 BPM 的滴答轨必须报 stable"
    assert 100 <= steady["median_tempo_bpm"] <= 145, steady["median_tempo_bpm"]
    assert not steady["tempo_changes"], "恒速轨不得编造速度变化"
    assert 55 <= min(steady["tempo_candidates_bpm"]) and max(steady["tempo_candidates_bpm"]) >= 120, \
        "倍速歧义候选必须保留"


def test_meter_is_unknown():
    out = rhythm.rhythm_timeline(_click_track(100.0, 30.0), SR, {"analysis": {}})
    assert out["meter"] == "unknown", "拍号没有小节强拍证据时不许猜"


# ---------------------------------------------------------------- profiles

def _curves_from_levels(levels: list[float], seconds_each: float = 10.0, interval: float = 0.5):
    n = int(seconds_each * 1 / interval)
    times = [round((i + 0.5) * interval, 3) for i in range(n * len(levels))]
    db, rate, low, bright, flux = [], [], [], [], []
    for lv in levels:
        db += [lv] * n
        rate += [2.0] * n
        low += [0.3] * n
        bright += [1000.0] * n
        flux += [0.1] * n
    return {"time_seconds": times, "rms_db": db, "onset_rate_hz": rate,
            "low_band_ratio": low, "spectral_brightness": bright, "spectral_flux": flux}


def test_rising_profile_detected_on_level_axis():
    curves = _curves_from_levels([-18.0, -15.0, -12.0, -9.0])
    segs = [{"start": i * 10.0, "end": (i + 1) * 10.0, "group": "ABCD"[i]} for i in range(4)]
    out = dprof.summarize_dynamic_profiles(curves, segs, [], {"analysis": {}})
    rising = [p for p in out["profiles"] if p["profile"] == "逐段增加"]
    assert rising and rising[0]["axis"] == "level_db" and rising[0]["evidence"]


def test_flat_axes_reported_individually_not_silently():
    curves = _curves_from_levels([-14.0] * 4)
    segs = [{"start": i * 10.0, "end": (i + 1) * 10.0, "group": "AB"[i % 2]} for i in range(4)]
    out = dprof.summarize_dynamic_profiles(curves, segs, [], {"analysis": {}})
    flat = [p for p in out["profiles"] if p["profile"] == "全曲平稳"]
    assert {p["axis"] for p in flat} == set(dprof.AXES), "每条平的轴都要单独声明，不能沉默"


def test_ending_stop_needs_both_axes_and_lowers_score():
    base = _curves_from_levels([-12.0, -12.0, -12.0, -12.0])
    times = base["time_seconds"]
    cut = int(len(times) * 0.9)
    for i in range(cut, len(times)):
        base["onset_rate_hz"][i] = 0.0
        base["rms_db"][i] = -60.0
    out = dprof.summarize_dynamic_profiles(base, [], [], {"analysis": {}})
    stop = [p for p in out["profiles"] if p["profile"] == "结尾骤停"]
    assert stop, "起音与电平同时收掉应报骤停"
    weak = [p for p in out["profiles"] if p["profile"] == "结尾骤停(弱证据)"]
    if weak:
        assert weak[0]["score"] < 0.5 and "截断" in " ".join(weak[0]["evidence"])


# ---------------------------------------------------------------- description

def test_description_never_invents_chorus_from_unknown_labels():
    dyn = {"events": [{"time": 10.0, "type": "axis-drop", "axis": "onset_rate_hz",
                       "evidence": ["axis:onset_rate_hz"]}]}
    segs = [{"start": 0.0, "end": 20.0, "group": "A", "label": "unknown"},
            {"start": 20.0, "end": 40.0, "group": "B", "label": "unknown"}]
    out = description.describe(dyn, {}, {"status": "unavailable", "reason": "mono"},
                               {"status": "ready", "median_tempo_bpm": 120,
                                "tempo_stable": True, "tempo_candidates_bpm": [60, 120, 240],
                                "tempo_changes": []},
                               {}, segs, {"analysis": {}})
    text = out["text"]
    assert "副歌" not in text, "标签全 unknown 时不得臆造副歌"
    assert "A/B" in text or "重复归为" in text
    assert "不可用" in text, "空间模块不可用必须如实说明"
    assert "维度" in text, "axis 事件要出现在维度变化一句里"
    assert out["meta"]["deterministic"] is True


def test_description_reports_dual_mono_honestly():
    spatial_ready = {"status": "ready",
                     "curves": {"lr_correlation": [0.98] * 20, "side_energy_ratio": [0.02] * 20}}
    out = description.describe({}, {}, spatial_ready, {}, {}, [], {"analysis": {}})
    assert "双单声道" in out["text"]
    assert "更有力量" not in out["text"], "禁止把亮度/宽度/密度合并成空泛的力量词"


def test_dynamics_axis_events_fire_on_flat_level():
    """电平恒定但起音密度变化：level 事件为零，axis 事件照报（验收样本 1/2）。"""
    levels = [-12.0] * 4
    curves = _curves_from_levels(levels)
    rate = [0.2] * 20 + [4.0] * 20 + [0.2] * 20 + [4.0] * 20
    # grid 内部键名：onset_rate（见 dynamics._grid_key 的映射）
    curves["onset_rate"] = rate
    events = dynamics.detect_axis_changes(curves, {"analysis": {}})
    level_events = [e for e in events if e["type"] in ("build", "peak", "drop", "pull-back", "release")]
    axis_events = [e for e in events if e["type"].startswith("axis-")]
    assert not level_events, "电平恒定时不得报电平事件"
    assert axis_events, "起音率变化必须由维度事件报出"
    assert any(e["axis"] == "onset_rate_hz" for e in axis_events)
