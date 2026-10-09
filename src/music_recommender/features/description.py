"""Deterministic Chinese description layer: structured data -> 制作描述.

Rules (expansion doc section 12):
  - every sentence must be traceable to a measured field; templates are fixed,
    no language model, no invention;
  - local beat/onset data writes "速度基本稳定，第二段起音活动增加" — never
    "加速" when only the onset density rose;
  - spatial claims only from the spatial module, level claims only from the
    level curve; "明亮、宽、密集、响" are never merged into "更有力量";
  - unknown sections stay "A/B"; "副歌" only when the label source says so;
  - realtime loopback sources must be labelled `system_loopback` and can never
    become per-track ground truth (expansion doc section 11).
"""
from __future__ import annotations


def describe(dyn: dict, texture: dict, spatial: dict, rhythm: dict,
             profiles: dict, segments: list[dict], cfg: dict | None = None) -> dict:
    lines: list[str] = []

    # ---- 结构与重复 ----
    if segments:
        groups: list[str] = []
        named = 0
        for seg in segments:
            g = seg.get("group", "")
            if g and g not in groups:
                groups.append(g)
            if seg.get("label") not in (None, "", "unknown"):
                named += 1
        if groups:
            head = f"全曲分为 {len(segments)} 段，重复归为 {len(groups)} 组（{''.join(groups)}，同组即重复）。"
            if named == 0:
                head += " 段落功能命名证据不足，一律用 A/B/C 字母指代，不臆造任何段落功能名称。"
            lines.append(head)

    # ---- 节奏与律动 ----
    if rhythm and rhythm.get("status") == "ready":
        med = rhythm.get("median_tempo_bpm") or 0
        cands = rhythm.get("tempo_candidates_bpm") or []
        if med:
            stab = "速度基本稳定" if rhythm.get("tempo_stable") else "局部速度有波动"
            line = f"{stab}（中位 {med:g} BPM"
            if len(cands) > 1:
                line += f"；存在半速/倍速歧义候选 {'/'.join(f'{c:g}' for c in cands)}，不作定论"
            line += "）。"
            for ch in (rhythm.get("tempo_changes") or [])[:2]:
                line += f" {ch['start']:.0f}-{ch['end']:.0f}s 局部速度偏向 {ch['tempo_bpm']:g} BPM。"
            lines.append(line)

    # ---- 动态事件（电平 + 维度） ----
    events = (dyn or {}).get("events") or []
    level_kinds = {"build", "peak", "drop", "pull-back", "release"}
    axis_names = {
        "onset_rate_hz": "起音活动", "onset_activity": "起音活动",
        "low_band_ratio": "低频比例", "spectral_brightness": "亮度", "spectral_flux": "音色变化",
    }
    level_bits: list[str] = []
    axis_bits: list[str] = []
    for ev in events:
        t = ev.get("type", "")
        time = ev.get("time", 0.0)
        if t in level_kinds:
            name = {"build": "电平堆叠", "peak": "电平峰值", "drop": "电平抽空",
                    "pull-back": "电平回落", "release": "电平释放"}.get(t, t)
            level_bits.append(f"{time:.0f}s {name}")
        elif t.startswith("axis-"):
            axis = axis_names.get(ev.get("axis", ""), ev.get("axis", "?"))
            direction = {"axis-rise": "上升", "axis-fall": "下降", "axis-drop": "骤降"}.get(t, t)
            axis_bits.append(f"{time:.0f}s {axis}{direction}")
    if level_bits:
        lines.append("电平走向：" + "、".join(level_bits[:4]) + ("等" if len(level_bits) > 4 else "") + "。")
    if axis_bits:
        lines.append("维度变化（与电平无关）：" + "、".join(axis_bits[:4]) + ("等" if len(axis_bits) > 4 else "") + "。")

    # ---- 音色纹理（模块自己的曲线说话） ----
    tex = texture or {}
    if tex.get("spectral_flatness") and tex.get("spectral_bandwidth"):
        flat = tex["spectral_flatness"]
        lo, hi = min(flat), max(flat)
        if hi - lo > 0.02:
            lines.append(f"频谱平坦度在 {lo:.2f}-{hi:.2f} 间变化（偏噪声性↔偏音调性），"
                         f"谱带宽 {min(tex['spectral_bandwidth']):.0f}-{max(tex['spectral_bandwidth']):.0f} Hz。")
        else:
            lines.append(f"频谱平坦度全曲近似恒定（{lo:.2f}-{hi:.2f}），音色底色稳定。")

    # ---- 空间（只从 spatial 模块说话） ----
    spa = spatial or {}
    if spa.get("status") == "ready":
        corr = [v for v in (spa["curves"].get("lr_correlation") or []) if v is not None]
        side = [v for v in (spa["curves"].get("side_energy_ratio") or []) if v is not None]
        if corr and side:
            import numpy as np
            mc, ms = float(np.median(corr)), float(np.median(side))
            if mc > 0.9 and ms < 0.1:
                lines.append(f"左右高度相关（中位相关度 {mc:.2f}，side 占比 {ms:.2f}）：接近双单声道内容。")
            else:
                lines.append(f"左右差异存在（中位相关度 {mc:.2f}，side 能量占比 {ms:.2f}）；"
                             "这是可测代理量，不代表混响深度或乐器前后。")
    elif spa.get("status") == "unavailable":
        lines.append(f"空间分析不可用（{spa.get('reason', 'unknown')}），不输出任何左右判断。")

    # ---- 动态轮廓 ----
    profs = (profiles or {}).get("profiles") or []
    if profs:
        bits = []
        for p in sorted(profs, key=lambda x: -x.get("score", 0))[:4]:
            iv = p.get("interval") or [0, 0]
            bits.append(f"{p['profile']}（{p.get('axis_label', p.get('axis', ''))}，{iv[0]:.0f}-{iv[1]:.0f}s）")
        lines.append("整体轮廓组合：" + " + ".join(bits) + "。")

    return {
        "lines": lines,
        "text": " ".join(lines),
        "meta": {
            "generator": "template-v1",
            "deterministic": True,
            "notes": ["描述全部由结构化数据模板生成，可逐句追溯到测量字段；未引入语言模型"],
        },
    }
