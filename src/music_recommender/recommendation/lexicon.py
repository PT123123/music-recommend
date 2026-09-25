"""One shared lexicon: sqlite feature column <-> query field names <-> Chinese words.

Everything that needs to say "this dimension, expressed as a human word" reads from
here, so the three consumers cannot drift:
  * `category.DIM_MAP`      - query field -> column
  * `discovery`             - auto category names from cluster centroids
  * `text_query`            - free Chinese text -> query fields

Honesty rule (ADR-1): only columns that are actually computed from local audio may
appear. A word we cannot back with a measurement is not added here; it is reported
as an unmatched term instead.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimSpec:
    col: str
    fields: tuple[str, ...]
    label: str
    high: tuple[str, ...]
    low: tuple[str, ...]
    # FEATURE_GROUPS name this dimension belongs to
    group: str = ""


def _d(col: str, fields: tuple[str, ...], label: str, high: tuple[str, ...],
       low: tuple[str, ...], group: str = "") -> DimSpec:
    return DimSpec(col=col, fields=fields, label=label, high=high, low=low, group=group)


# Each entry: measured column -> (query field names, dimension label,
# high-side words, low-side words). The FIRST word of each side is the canonical one
# used when naming a discovered cluster; the rest are synonyms the free-text parser
# accepts. Synonyms must stay acoustically literal: mood words we cannot back with a
# measurement ("治愈", "伤感", "高级感") are deliberately absent so the parser reports
# them as unanswerable instead of quietly mapping them onto something nearby.
LEXICON: tuple[DimSpec, ...] = (
    # ---- energy ----
    _d("rms_mean", ("energy", "loudness", "rms_mean"), "响度",
       ("高能量", "响", "炸", "燃", "带劲", "热血"),
       ("轻柔", "低能量", "安静", "宁静", "平和"), "energy"),
    _d("rms_std", ("energy_variation", "rms_std"), "能量波动",
       ("起伏大", "段落对比强"), ("平稳", "一条线"), "energy"),
    _d("dynamic_range", ("dynamic_range", "dynamics"), "动态范围",
       ("动态大", "动态对比大"), ("动态平直",), "energy"),
    _d("crest_factor", ("crest",), "峰值",
       ("峰值突出",), ("峰值被压平",), "energy"),
    # ---- timbre ----
    _d("spectral_centroid_mean", ("brightness", "timbre_brightness", "spectral_centroid_mean"),
       "音色明暗", ("明亮", "通透", "清亮", "清脆"), ("暗沉", "阴暗", "闷"), "timbre"),
    _d("spectral_bandwidth_mean", ("bandwidth",), "频谱宽度",
       ("频谱宽", "声场铺得开"), ("频谱集中", "声场窄"), "timbre"),
    _d("spectral_flatness_mean", ("flatness", "noisiness"), "噪声成分比",
       ("气声感", "沙哑质感", "毛刺感"), ("纯净乐音",), "timbre"),
    _d("spectral_contrast_mean", ("contrast",), "音色对比",
       ("音色层次分明",), ("音色混为一体",), "timbre"),
    _d("spectral_flux_mean", ("flux",), "频谱变化",
       ("音色变化频繁",), ("音色稳定",), "timbre"),
    _d("zcr_mean", ("zcr", "hissiness"), "过零率",
       ("偏噪偏齿音", "沙沙感"), ("圆润",), "timbre"),
    # ---- rhythm ----
    _d("bpm", ("tempo", "bpm"), "速度",
       ("快节奏", "快速", "快歌"), ("慢速", "慢节奏", "慢歌", "舒缓"), "rhythm"),
    _d("onset_density", ("rhythm_intensity", "onset_density"), "节奏密度",
       ("节奏密集", "律动密", "拍子密", "鼓点密", "节奏感强", "节奏感"), ("节奏稀疏", "节奏感弱"), "rhythm"),
    _d("danceability", ("danceability", "groove"), "可舞性",
       ("律动舞曲感", "弹跳感", "适合跳舞"), ("不弹跳",), "rhythm"),
    _d("beat_consistency", ("beat_consistency", "steadiness"), "节拍稳定度",
       ("节拍稳定", "卡点准"), ("节拍自由",), "rhythm"),
    _d("tempo_variance", ("tempo_variance",), "速度波动",
       ("速度波动", "变速"), ("速度恒定",), "rhythm"),
    # ---- harmony ----
    _d("chord_change_rate", ("chord_change_rate", "harmonic_activity"), "和声变化率",
       ("和声变化频繁", "和声复杂"), ("和声简单循环", "四个和弦循环"), "harmony"),
    _d("harmonic_rhythm", ("harmonic_rhythm",), "和声节奏",
       ("和声推进快",), ("和声停留长",), "harmony"),
    _d("dissonance_mean", ("dissonance", "tension"), "不协和度",
       ("不协和张力", "紧张感", "刺耳"), ("和谐悦耳", "顺耳"), "harmony"),
    # ---- melody ----
    _d("pitch_mean", ("pitch", "pitch_mean"), "整体音区",
       ("整体音区高", "音调高"), ("整体音区低", "音调低"), "melody"),
    _d("pitch_range", ("melody_range", "pitch_range"), "旋律跨度",
       ("旋律跨度大", "音域宽"), ("旋律平缓", "平铺直叙"), "melody"),
    _d("pitch_variance", ("pitch_variance",), "旋律起伏",
       ("旋律起伏大", "旋律跌宕"), ("旋律平稳",), "melody"),
    _d("high_pitch_ratio", ("pitch_high", "high_pitch_ratio"), "高音成分",
       ("高音成分多", "高音区为主"), ("高音成分少",), "melody"),
    _d("low_pitch_ratio", ("pitch_low", "low_pitch_ratio"), "低音成分",
       ("低频成分厚", "低音重"), ("低音成分少",), "melody"),
    _d("mid_pitch_ratio", ("pitch_mid",), "中频成分",
       ("中频为主", "厚实"), ("中频占比低",), "melody"),
    # ---- vocal (percentiles only exist on tracks the detector believes have vocals) ----
    _d("vocal_ratio", ("vocal_presence", "vocal_ratio"), "人声占比",
       ("人声突出", "人声为主", "贴耳"), ("人声稀薄", "伴奏为主", "人声远"), "vocal"),
    _d("vocal_intensity", ("vocal_intensity",), "演唱强度",
       ("唱得用力", "爆发", "嘶吼", "吼"), ("轻声细语", "耳语感", "轻声"), "vocal"),
    _d("vocal_pitch_mean", ("vocal_pitch", "vocal_pitch_mean"), "人声音区",
       ("高音人声", "高亢", "高音"), ("低沉人声", "低沉", "低音人声"), "vocal"),
    _d("vocal_pitch_range", ("vocal_range", "vocal_pitch_range"), "人声音域跨度",
       ("音域跨度大", "转音多"), ("音域窄", "说话感", "念白感"), "vocal"),
    _d("vocal_pitch_variance", ("vocal_pitch_variance",), "人声音高起伏",
       ("人声起伏大", "转音丰富"), ("人声平直", "念白"), "vocal"),
    # ---- instrumentation ----
    _d("bass_energy_ratio", ("bass_energy", "bass_energy_ratio"), "低频能量比",
       ("低音饱满", "重低音", "低频足", "贝斯重"), ("低音轻", "少低音"), "instrumentation"),
    _d("drum_energy_ratio", ("drum_energy", "drum_energy_ratio"), "鼓能量比",
       ("鼓点强", "鼓声大"), ("鼓点弱", "少鼓"), "instrumentation"),
    # ---- structure (percentiles exist only when the structure module produced them) ----
    _d("chorus_energy", ("chorus_energy",), "副歌能量",
       ("副歌能量高", "副歌炸"), ("副歌不收放",), "structure"),
    _d("chorus_repeat_count", ("chorus_repeat",), "副歌重复次数",
       ("反复洗脑", "循环多"), ("结构少重复",), "structure"),
)

BY_COL: dict[str, DimSpec] = {d.col: d for d in LEXICON}

# query field -> column, derived so config keys and code can never disagree
DIM_MAP: dict[str, str] = {f: d.col for d in LEXICON for f in d.fields}
# column -> the canonical field name to write back into a category query
FIELD_BY_COL: dict[str, str] = {d.col: d.fields[0] for d in LEXICON}

# column -> {field: canonical word for the high side / low side}, used by discovery
WORDS_HIGH: dict[str, str] = {d.col: d.high[0] for d in LEXICON}
WORDS_LOW: dict[str, str] = {d.col: d.low[0] for d in LEXICON}
LABELS: dict[str, str] = {d.col: d.label for d in LEXICON}

# All words the text parser may see, longest-first so "低沉人声" wins over "人声".
TEXT_TERMS: dict[str, tuple[str, int]] = {}
for _d_spec in LEXICON:
    for _w in _d_spec.high:
        TEXT_TERMS[_w] = (_d_spec.col, +1)
    for _w in _d_spec.low:
        TEXT_TERMS[_w] = (_d_spec.col, -1)
_SORTED_TERMS: tuple[str, ...] = tuple(sorted(TEXT_TERMS, key=len, reverse=True))

# Tag-derived (NOT audio-derived) hard-filter vocabulary. Values are the file's own
# embedded claim (ADR-14); a track without the tag can never match, and the engine
# reports that count instead of guessing.
LANGUAGE_TERMS: dict[str, str] = {
    "中文": "zh", "国语": "zh", "华语": "zh", "国粤": "zh", "粤语": "yue", "粤": "yue",
    "英文": "en", "英语": "en", "日文": "ja", "日语": "ja", "韩文": "ko", "韩语": "ko",
    "纯音乐": None, "器乐": None,
}
# Genre words people type constantly. Two honest paths, in this order:
#   1. the file carries a genre tag -> exact `genre_is` hard filter;
#   2. no tag in this library -> the acoustic approximation below, reported with a
#      caveat so the answer never claims to know the genre. The approximation is the
#      same dimension set the matching preset category in categories.yaml uses.
# Nothing here pretends to detect a scene, an era or a mood label.
GENRE_TERMS: dict[str, tuple[str, dict[str, float]]] = {
    "摇滚": ("rock", {"energy": 0.80, "rhythm_intensity": 0.70, "drum_energy": 0.70, "brightness": 0.60}),
    "说唱": ("hip", {"vocal_pitch_range": 0.15, "rhythm_intensity": 0.85, "vocal_presence": 0.80}),
    "嘻哈": ("hip", {"vocal_pitch_range": 0.15, "rhythm_intensity": 0.85, "vocal_presence": 0.80}),
    "电子": ("electronic", {"drum_energy": 0.80, "beat_consistency": 0.80, "tempo": 0.70}),
    "舞曲": ("dance", {"danceability": 0.85, "drum_energy": 0.75, "tempo": 0.75}),
    "民谣": ("folk", {"energy": 0.30, "vocal_presence": 0.70, "bandwidth": 0.40}),
    "古典": ("classical", {"drum_energy": 0.15, "dynamic_range": 0.70, "flatness": 0.20}),
    "爵士": ("jazz", {"dissonance": 0.65, "beat_consistency": 0.35, "brightness": 0.45}),
    "R&B": ("r&b", {"vocal_presence": 0.80, "tempo": 0.45, "bass_energy": 0.65}),
    "r&b": ("r&b", {"vocal_presence": 0.80, "tempo": 0.45, "bass_energy": 0.65}),
    "金属": ("metal", {"energy": 0.90, "brightness": 0.70, "rhythm_intensity": 0.85, "dissonance": 0.70}),
    "朋克": ("punk", {"tempo": 0.85, "energy": 0.85, "beat_consistency": 0.70}),
    "蓝调": ("blues", {"dissonance": 0.45, "tempo": 0.35, "vocal_presence": 0.70}),
    "乡村": ("country", {"vocal_presence": 0.75, "energy": 0.45, "brightness": 0.60}),
    "国风": ("chinese", {"brightness": 0.55, "vocal_presence": 0.75, "bandwidth": 0.45}),
    "古风": ("chinese", {"brightness": 0.55, "vocal_presence": 0.75, "bandwidth": 0.45}),
    "流行": ("pop", {"energy": 0.60, "vocal_presence": 0.75, "chorus_energy": 0.70}),
}
# "女声 / 男声" is the other thing people type constantly, and it is the one word here
# we genuinely cannot measure: no local feature in this project identifies a singer's
# gender. What we can measure is the register the word is usually a shorthand for, so
# the term maps to it and is reported as an approximation (same contract as a genre
# proxy), never as a detected attribute.
VOCAL_PROXY_TERMS: dict[str, tuple[str, dict[str, float]]] = {
    "女声": ("人声音区偏高", {"vocal_pitch": 0.85, "vocal_range": 0.75}),
    "女生": ("人声音区偏高", {"vocal_pitch": 0.85, "vocal_range": 0.75}),
    "女音": ("人声音区偏高", {"vocal_pitch": 0.85, "vocal_range": 0.75}),
    "男声": ("人声音区偏低", {"vocal_pitch": 0.18, "vocal_range": 0.45}),
    "男生": ("人声音区偏低", {"vocal_pitch": 0.18, "vocal_range": 0.45}),
    "男音": ("人声音区偏低", {"vocal_pitch": 0.18, "vocal_range": 0.45}),
}


def name_from_deviations(devs: list[tuple[str, float]], max_terms: int = 3) -> str:
    """Turn [(column, signed percentile deviation)] into a Chinese category name."""
    parts = []
    for col, dev in devs[:max_terms]:
        parts.append(WORDS_HIGH.get(col) if dev > 0 else WORDS_LOW.get(col))
    parts = [p for p in parts if p]
    return "·".join(parts) if parts else "无明显倾向"


def match_terms(text: str) -> list[tuple[str, str, int]]:
    """Greedily match lexicon words in free text -> [(word, column, direction)].

    Longest match wins and consumed spans are removed, so "低沉人声" is one term
    rather than "低沉" + "人声".
    """
    out: list[tuple[str, str, int]] = []
    rest = text or ""
    for word in _SORTED_TERMS:
        idx = rest.find(word)
        if idx < 0:
            continue
        col, direction = TEXT_TERMS[word]
        out.append((word, col, direction))
        rest = rest[:idx] + "　" * len(word) + rest[idx + len(word):]
    out.sort(key=lambda w: (text or "").find(w[0]))
    return out
