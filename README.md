# Local Music MIR Recommendation System

完全离线的本地音乐 **MIR 特征提取 + 内容驱动推荐引擎**。从本地 MP3/WAV 音频本身推导
Music Representation / Music Space，并对外提供 **歌曲相似推荐**、**固定类别推荐** 和
**带时间衰减的动态 Feed**。推荐内容来自音频分析，而非外部平台人工标签。

> 当前实现范围:**Phase 1–7 全部落地并端到端验证**。类别 / Feed / 评测 / 高阶特征均已实现;
> 凡本地无法可靠得到的属性(性别 / 外部热度 / 语言 / 命名乐器 / genre / arousal)一律留空或标 `unknown`,**绝不伪造**。见下文「实现进度」。

---

## 实现进度(对照规格 7 个阶段)

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | MP3/WAV → 标准化 → RMS/ZCR/频谱/MFCC/BPM/Chroma/Key → SQLite → PCA Embedding → FAISS → 歌曲到歌曲 | ✅ 已完成 |
| Phase 2 | 和弦估算(转调不变)、旋律音高/音程、人声存在/音高、配器能量;并入 Metadata Re-ranking | ✅ 已完成(估算字段标 `estimated`) |
| Phase 3 | 分段 / 能量曲线(64bin) / 副歌检测 / 结构相似(曲线相关 + 段序列) + `track_segments` 表 | ✅ 已完成 |
| Phase 4 | 固定类别:`categories.yaml` → Music Space 区域(hard_filter + soft ranking),`/v1/recommend/category` + CLI | ✅ 已完成 |
| Phase 5 | 动态 Feed:指数时间衰减 short/medium/long 兴趣、反馈、novelty/diversity(MMR)、近期去重、冷启动多样性 | ✅ 已完成 |
| Phase 6 | Local HTTP API:health/tracks/categories/similar/scan/index + category/feed(next·feedback·state·reset)/evaluate | ✅ 已完成 |
| Phase 7 | 人工评测:pairwise 标签记录 + 导出 `seed,recommendation,label` CSV + `/v1/evaluate` | ✅ 已完成 |

> **诚实性 & 性能**:和弦/旋律/人声/结构均为 librosa/CPU **估算**(带 `estimate_flags` 与置信度),接口可替换为专用模型。Phase 2–3 引入 `pyin`+`hpss` 后,**真实曲库实测 43.1 秒/首**(单进程;其中 pyin 28.7 秒),合成 4 秒短片才是十几秒一首——早期文档写的"每首数秒"是短片结论,已改正。大库抽取必须并行 + 增量(`--workers 6` 下 93 首 ≈22 分钟,重扫 0 秒),或换更便宜的 f0/CQT 方案。查询侧已优化到 **20 毫秒量级**并可脱离 Python 运行,见「性能与可移植性」。

设计遵循规格「实现原则」:模型可替换(`AudioEmbedder` 接口)、SQLite 为事实来源、FAISS
仅为可重建缓存、不满足字段不伪造、配置化权重与阈值。

---

## 环境要求

- **Python 3.10 – 3.12**(推荐 3.12)。⚠️ **不要用 Python 3.14** —— librosa / faiss-cpu /
  scikit-learn 目前无对应 wheel。
- **FFmpeg**(用于 MP3 解码标准化)。Windows 可用 `winget install Gyan.FFmpeg`。
  若不在 PATH,librosa 会通过 `audioread` 回退解码(可能给出 deprecation 警告,功能正常)。
- **可选:Rust(仅 `rust/musicspace` 端口需要)**,`cargo` 稳定版即可。该 crate 只依赖 std,
  不联网拉 crates.io;主引擎不需要它。

### 安装依赖(使用 uv,推荐)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# 可选:开发/测试
uv pip install --python .venv/Scripts/python.exe pytest httpx
```

或用普通 pip:

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

## Embedding / 模型配置

MVP 使用 **轻量占位方案**:`librosa 统计特征 → StandardScaler → PCA(16~32 维) → L2 归一化`,
**不下载任何大模型**,纯 CPU。配置见 `configs/config.yaml` 的 `embedding` 段。

想要更高语义召回,实现 `music_recommender/embedding/base.py` 的 `AudioEmbedder` 接口
(如接入 PANNs / MERT),在 `config.yaml` 里切换 `backend` 即可,核心代码不依赖具体模型。

## 目录说明

```
configs/     config.yaml(路径/音频/embedding/检索/feed 衰减权重) weights.yaml(推荐分组权重) categories.yaml(类别定义)
docs/        ADR.md(设计决策) CHANGELOG.md(Phase 1-7 交付) PERFORMANCE.md(性能与瓶颈) PORTING.md(移动端可行性)
scripts/     CLI 入口(见下)
src/music_recommender/
  preprocess/  音频标准化 (ffmpeg / librosa) + 内嵌 tag 读取 (mutagen)
  features/    MIR 特征提取 (acoustic / rhythm / harmony / melody / vocal / instruments / structure)
  embedding/   AudioEmbedder 抽象 + PCA 占位实现
  database/    SQLite schema + repository(事实来源)+ behavior(互动/兴趣快照/评测)
  vector_store/FAISS 索引(可重建缓存)
  recommendation/ Music Space 归一化 + 推荐核心(相似/类别/Feed 共用)+ sequence_sim + category
  interest/    InterestModel(时间衰减兴趣)+ FeedEngine(无脑流)
  api/         FastAPI 服务器
data/        normalized_audio/  sqlite/  faiss/  test_music/(合成测试音频)
rust/musicspace/ 在线打分路径的 std-only Rust 端口(移动端可行性验证,见 docs/PORTING.md)
logs/        pipeline.log  errors.log
tests/       pytest 套件 + 端到端校验
```

## 快速开始(端到端)

```bash
# 1) 生成合成测试音频(无需真实曲库即可验证管线)
python scripts/make_test_audio.py data/test_music

# 2) 一条龙:扫描 + 抽特征 + 训练 embedding + 建 FAISS
python scripts/scan_library.py data/test_music

#   或分步:
python scripts/extract_features.py data/test_music   # 仅抽特征入库
python scripts/build_embeddings.py                   # 拟合 PCA + 存 embedding
python scripts/build_faiss.py                        # 由 embedding 建 FAISS

# 3) 真实曲库(建议开并行;长曲目按能量最密窗口分析,见 docs/PERFORMANCE.md)
python scripts/scan_library.py "D:/Music/我的歌单" --workers 6

# 4) 歌曲 → 相似歌曲
python scripts/recommend.py data/test_music/bright_pop_1.wav --limit 5
python scripts/recommend.py data/test_music/bright_pop_1.wav --json
python scripts/recommend.py --track-id <id> --limit 10
python scripts/recommend.py --category high_female_vocal            # 固定类别
```

> 分析窗口:`configs/config.yaml` 的 `audio.max_analysis_seconds`(默认 45)会取每首曲子
> **能量最密的连续片段**来抽特征(而不是前奏),把 4 分钟单曲从 ≈172 秒压到 ≈20 秒级;
> `duration` 仍记录整曲真实时长。设 `0` 表示整曲分析。`workers` 只影响抽取阶段。

> **重扫是增量的**:文件大小 + mtime 未变且已抽过特征就直接跳过(实测第二次扫描 0.0 秒)。
> 改了特征算法需要全量重算时加 `--force`。`track_id` 由文件内容哈希得到,所以重命名或
> 挪目录不会丢失该曲的历史行为数据。
>
> **标题/歌手/专辑/年份**来自音频文件**自带的 tag**(mutagen),存库时带 `meta_source='tag'`
> 标记来源;没有 tag 就留空,**不从文件名猜**。这些字段只是给播放器显示用的事实,不参与
> 内容推荐,也不会被当成音频分析结果。

输出示例:
```
Seed: e5019fc217166ab2
1. bright_pop_3.wav   0.686   整体音色/听感接近, 频谱音色接近
2. bright_pop_2.wav   0.678   ...
```

推荐分数已按规格归一化到 [0,1]:各分组相似度先 [0,1] 化再按 `weights.yaml` 加权,并只对该
pair 实际可算的分组重新归一权重。`reasons` 来自真实计算结果(见「推荐解释」)。

## HTTP API

```bash
python scripts/run_server.py --port 8000
# 打开 http://127.0.0.1:8000/docs  (FastAPI 自动 OpenAPI 文档)
```

已实现端点(全部返回 200):

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/v1/health` | 健康检查 |
| GET  | `/v1/tracks/{track_id}` | 返回完整 Music Representation(去除二进制数组) |
| GET  | `/v1/categories` | 固定类别列表(来自 `categories.yaml`) |
| POST | `/v1/recommend/similar` | 歌曲到相似歌曲;body `{"track_id": "..."}` 或 `{"file_path": "..."}`,`limit` |
| POST | `/v1/recommend/category` | 固定类别推荐:`{"category_id":"high_energy","limit":30}` 或参数化 `{"query":{"energy":0.85,"brightness":0.2},"hard_filter":{"has_vocal":false}}` |
| POST | `/v1/feed/next` | 动态 Feed 下一批:`{"user_id":"local-user","limit":20,"exclude_track_ids":[...]}` |
| POST | `/v1/feed/feedback` | 反馈事件:`{"track_id":"...","event":"like\|dislike\|skip\|play\|complete\|replay\|partial\|impression","completion_ratio":0.9}` |
| GET  | `/v1/feed/state` | 当前 short/medium/long 兴趣摘要(可解释) |
| POST | `/v1/feed/reset` | 清空兴趣:`?scope=short\|medium\|long\|all`(默认 short,不动长期) |
| POST | `/v1/evaluate` | 记录人工评测标签:`{"seed_track_id":..,"recommended_track_id":..,"label":1}` |
| POST | `/v1/library/scan` | 扫描目录并端到端入库+建索引 `{"root":"D:/Music","recursive":true,"workers":6,"force":false}` |
| POST | `/v1/tracks/index` | 单文件入库 `{"file_path":"D:/Music/a.mp3"}` |

三个推荐端点(`/v1/recommend/similar`、`/v1/recommend/category`、`/v1/feed/next`)的每条结果都带
一个 `display` 块 `{title, artist, album, file_path, meta_source}`,播放器一次请求即可渲染列表;
`meta_source` 用来区分「文件自带」与「音频分析」,前者的值可能为 `null`(该文件没有 tag)。

示例:
```bash
curl -s http://127.0.0.1:8000/v1/health
curl -s -X POST http://127.0.0.1:8000/v1/recommend/similar \
     -H "Content-Type: application/json" \
     -d '{"track_id":"e5019fc217166ab2","limit":5}'
curl -s -X POST http://127.0.0.1:8000/v1/recommend/category \
     -H "Content-Type: application/json" -d '{"category_id":"high_energy","limit":10}'
curl -s -X POST http://127.0.0.1:8000/v1/feed/feedback \
     -H "Content-Type: application/json" -d '{"track_id":"e5019fc217166ab2","event":"like"}'
curl -s -X POST http://127.0.0.1:8000/v1/feed/next \
     -H "Content-Type: application/json" -d '{"limit":10}'
```

## Python API

```python
from music_recommender import MusicLibrary, Recommender

lib = MusicLibrary()
lib.index("data/test_music")      # 扫描 + 抽特征(Phase 1-3);真实曲库用 workers=6, force=...
lib.rebuild_embeddings()          # 拟合 PCA + 存 embedding
lib.build_faiss()                 # 建 FAISS

rec = Recommender()
for r in rec.similar(track_id="...", limit=10):
    print(r.track_id, r.score, r.reasons)

items, meta = rec.category("high_energy", limit=20)      # 固定类别
rec.feedback("local-user", track_id="...", event="like") # 记录行为
feed = rec.feed_next("local-user", limit=10)             # 动态 Feed(时间衰减)
print(rec.feed_state("local-user"))                      # 可解释兴趣摘要
```

HTTP 与 Python API 共用同一 `Recommendation Core`(规格 42/53):歌曲相似 / 类别 / Feed 区别
只在「查询向量来源」——分别是种子曲、类别区域、时间衰减兴趣向量。

CLI 补充(Feed / 评测):
```bash
python scripts/feed.py feedback --track-id <id> --event like
python scripts/feed.py next --limit 10
python scripts/feed.py state
python scripts/feed.py reset --scope short
python scripts/evaluate.py                       # 交互式打标签
python scripts/evaluate.py --export data/evaluations.csv
```

## 推荐解释 / Music Space

`recommendation/space.py` 把每个标量特征在**全库内**转成经验分位(CDF → [0,1]),因此
相似度是「这首歌在本地库中处于什么位置」的相对度量(对应规格 14「极端程度」)。分组:
energy / timbre / rhythm / harmony / melody / vocal / instrumentation / structure(Phase 1–3),
外加 embedding(FAISS 余弦)。和弦/旋律用**转调不变序列**做编辑距离+n-gram+Jaccard+转移相似
(规格 §10)。`reasons` 仅当对应分组真实相似度 ≥ 阈值时给出,不编造「因为你最近听过…」(规格 59)。

## 测试

```bash
# 端到端控制校验:每个种子最近邻应落在同一家族(合成数据按族构造)
python scripts/scan_library.py data/test_music   # 注意:写入的是 config 指向的那个库
python tests/verify_mvp.py                       # 期望 VERDICT: PASS(12/12)
#   它靠合成文件名判定家族;对着真实曲库的库跑会打印 SKIP,而不是给出错的结论。
#   跑完想回真实曲库:python scripts/scan_library.py "D:/Music/我的歌单" --workers 6

# 单元 + 集成测试(独立临时 DB / FAISS,不污染 data/)
python -m pytest tests/test_pipeline.py -q        # 管线 / embedding / FAISS / 推荐 / 评测
python -m pytest tests/test_scan_identity.py -q   # 扫描身份 / tag 读取 / 增量跳过 / 序列预算 / DSP 去重
python -m pytest tests/test_feed.py -q            # Feed 时间衰减 + 差分喜欢偏好
python -m pytest tests/test_query_perf.py -q      # 查询路径等价性:朴素参考实现逐位比对(rapidfuzz / 分组相似度 / MMR / 冷启动)
```

## 性能与可移植性

真实曲库 93 首 WAV 实测(`docs/PERFORMANCE.md` 有全过程与两次勘误):

| | 数值 |
|---|---|
| song→song / category / feed_next | **21.8 ms** / 3.8 ms / 3.3 ms |
| 离线抽取 | 43.1 秒/首(单进程,`pyin` 28.7 + `hpss` 5.9 占 80%);93 首 6 worker ≈ 22 分钟;重扫未变更曲库 0 秒 |
| 打分载荷 | **1560 字节/首** + embedding 96 字节(整行 11,231 字节,其余是抽取中间量,查询不读)→ 5000 首约 8.3 MB |

关于"能不能塞进手机":**在线打分路径已用 std-only Rust 重写并验证与 Python 排名逐位一致**(12 seed × 20 位次 = 240 slot,0 处顺序不一致、分数差 0,Rust 6.1 ms/查询);命令内置**反向对照**,故意改坏一个权重必须被抓出,否则判定不成立。**离线抽取不搬手机**——那是 librosa 的算法成本,不是语言成本。手机形态 = 桌面抽取、设备只查询。完整论证、边界与移植清单见 `docs/PORTING.md`。

```bash
PYTHONPATH=src python scripts/bench_portability.py --json                       # 载荷/算术实测 + 导出比对物
cd rust/musicspace && cargo run --release -- data/portability --verify          # 等价性 + 反向对照 + 计时
```

## 故障处理

| 现象 | 处理 |
|------|------|
| `pip install` 在 Python 3.14 找不到 faiss/librosa wheel | 改用 3.12:`uv venv --python 3.12` |
| MP3 读取失败 / `audioread` 警告刷屏 | 安装 FFmpeg 并加入 PATH |
| `n_fft too large` 警告 | 音频过短(测试用),真实歌曲不会出现,可忽略 |
| 只改了音频没重建 | 重新跑 `scan_library.py` 或 `/v1/library/scan` |
| 想换 embedding | 实现 `AudioEmbedder` 并改 `config.yaml: embedding.backend` |
| 删除 FAISS 后无法检索 | FAISS 仅为缓存,跑 `build_faiss.py` 由 SQLite 重建 |

## 模型 / 数据许可证

- 本仓库代码:实现自 MVP,无第三方模型权重。
- FFmpeg:LGPL/GPL(视构建)。
- 未来接入 PANNs / MERT 等需遵守其各自权重许可(接入前核对)。

## 设计文档

- `docs/ADR.md` — 关键设计决策(禁止伪造字段、Music Space 分位归一化、embedding 可替换、和弦转调不变、类别/Feed 共用核心、Feed 防坍缩、配置驱动、有界分析窗、内容稳定 track_id、tag 来源标注、**在线路径等价优化 + Rust 端口与反向对照(ADR-15)、Music Space 按库版本缓存(ADR-16)**)。
- `docs/CHANGELOG.md` — Phase 1–7 逐竖切交付清单 + 之后每一轮的实测修正。
- `docs/PERFORMANCE.md` — 实测耗时、复杂度、后续优化方向(含两次勘误的来龙去脉)。
- `docs/PORTING.md` — 移动端可行性:载荷/算术实测、std-only Rust 端口的逐位次等价验证、已知边界与移植清单。
