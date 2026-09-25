# Performance Notes

## 已知瓶颈:pyin 基频 + hpss 分解(CPU,离线)
`features/melody.py`(pyin f0)与 `features/extract.py`(hpss)是整条流水线主要开销。

## 真实曲库实测(本地歌单文件夹,93 首 WAV)
93 首 WAV / 3.7 GB,时长 76–433 秒(均值 231 秒)。

| 配置 | 结果 |
|------|------|
| 整曲分析(`max_analysis_seconds: 0`),单进程 | 单首 297 秒曲目 = **172 秒**抽特征 → 93 首约 **4 小时** |
| 45 秒分析窗 + 6 worker(去重前) | 93 首全部入库 = **约 32 分钟**(≈20.6 秒/首有效),0 失败 |
| 45 秒分析窗 + 6 worker(pyin/hpss 去重后) | 93 首入库 + embedding + FAISS = **约 22 分钟**,0 失败 |
| 重扫未变更曲库(`index()` 增量跳过) | 第二次扫描 **0.0 秒**(indexed=0, skipped=N) |
| song→song 查询(序列预算前,93 首库) | **266 秒** / 20 候选 ← 原缺陷 |
| song→song 查询(预算 120 token + 1 半音分桶) | 平均 **1.06 秒** |
| song→song 查询(预算 120 token + 2 半音分桶,最终配置) | 10 次平均 **0.55 秒**(最大 0.65 秒) |
| category 查询 / feed_next | **0.026 秒** / **0.099 秒** |

**勘误**:此行原写"加载后查询亚秒级",那是合成语料(4 秒短片、序列仅几十 token)上的结论,真实曲库实测 266 秒。已按真实数据改正,并记下修复位置(`features.max_sequence_tokens`)。

序列预算 + 分桶修正对成品库的副作用(93 首真实曲库,全部重新抽取后):旋律 token 数中位数 120(上限)、最大 120,修复前 2677–4015;interval_histogram 里 ±1 半音"音高变化"占比 **0.000**(14202 步全量,修复前 83–90%);`music.db` 从 2.4 MB 降到 **1.04 MB**(≈11.2 KB/首)。

单首抽取拆解(45 秒窗,单进程,3 首真实曲目 33.6 / 35.6 / 40.7 秒):`pyin` **28.7 秒** + `hpss` **5.9 秒** ≈ 43.1 秒里的 80%。去重前 pyin 调 2 次、hpss 调 3 次,这就是并行前后墙钟差异的来源。

`normalize`(解码整曲)≈13 秒/首仍是并行下的剩余固定成本,可改成 ffmpeg `-ss/-t` 只解码目标窗口。机器 22 核,worker 数仍可上调。

**分析窗策略**:`extract._analysis_window()` 不按开头切,而是用 1 秒块能量做滑动求和,取**能量最密集的 45 秒**——保证窗口覆盖副歌而非前奏,这样 `chorus_*` / `energy_curve` / 人声维度仍有代表性;`duration` 列始终保留整曲真实时长,窗口起止存于 `features_json.analysis_window`。

**worker 线程收敛**:`MusicLibrary.index(workers>1)` 会在起进程池前把 `OMP/OPENBLAS/NUMBA/MKL_NUM_THREADS` 设为 1,避免 N 个进程各开满线程互相争抢。

**合成语料(12 段 4 秒)**:4 秒短片低于任何窗口上限,行为与不分窗时完全一致。

## 数据规模相关
| 操作 | 复杂度 | 备注 |
|------|--------|------|
| 特征抽取 / 首 | O(T·hop) | pyin 是主项,hop 越长越快 |
| Music Space 分位 | O(N·log N) | 换库 / 加曲后需重算 |
| FAISS 检索 top-K | O(N·K) | IndexFlatIP;> ~10k 曲考虑换 IVF |
| Feed 打分 | O(K²) | MMR 多样性贪心,K=200 时 ~40k 内积 |
| 重排 | O(K) | 每对约 8 分组,分组内 O(L²) 序列距离,L ≤ `features.max_sequence_tokens`(此前 L 可达 4000+) |

## 后续可优化方向
1. **song→song 剩余 ~0.55 秒**:266 秒缺陷修掉后,93 首库单次查询仍约 0.55 秒,来源是每次调用重建 Music Space 分位 + 20 候选的纯 Python Levenshtein(120×120 每次)。可选:(a) 缓存按库版本失效的 Music Space;(b) 用 numpy 行向量改写动态规划或加 Udi Manber 前置界;(c) 把序列相似度搬到 Rust/C++。**离线批处理阶段(Rust/C++ 真正的收益点)是 `pyin`/`hpss` 那 80%**,不是查询。
2. **再加并行**:22 核机器上 `--workers` 仍可上调(12–16);剩余固定成本是 `normalize` 解码整曲,可改成 ffmpeg `-t/-ss` 只解码目标窗口,把 ≈13 秒/首也省掉。
3. **缓存 raw analysis 中间量**:把 chroma / hpss / f0 结果落盘(比 wav 小),重排调参时秒级重跑。
4. **换后端 embedding 时一次性重算**:改 `configs/config.yaml: embedding.backend` → 重跑 `build_embeddings` + `build_faiss`,SQL 主表不变。特征算法本身变更时用 `scan_library.py --force`(见 ADR-13)。
5. **测试加速**:临时库 fixture 复用同一份特征缓存,只重建 index。
6. **磁盘**:`normalize` 会为每首写出 44.1kHz 单声道 WAV,93 首占用 **约 1.9 GB**(`data/normalized_audio/`)。该目录是可重建缓存。注意 track_id 改内容哈希后(ADR-12)旧文件名的缓存全部成为孤儿,新旧两轮并存过一次;确认新库无误后可以把旧那批挪进 `.quarantine-<日期>/`。

## 元数据体积(决定能否常驻内存 / 是否需要查询层)
93 首真实曲库全量入库后实测:

| 量 | 值 |
|----|----|
| `music.db` 整库 | 1,044,480 字节 ≈ **11.2 KB/首**(序列预算前是 2.4 MB ≈ 26.5 KB/首) |
| `SELECT *` 读全部行 | **0.004 秒**(93 行) |
| FAISS 索引 + id 映射 | 9 KB + 1.8 KB;embedding 24 维 × 93 首 |
| song→song 单次 | **0.55–0.65 秒**(每次调用重算 Music Space 分位) |

结论:元数据体积小到一个数量级——整库常驻内存毫无压力,进程起来 4 毫秒就能读全表,所以**查询层不需要为"少取数据"设计**;真正需要处理的是抽特征阶段(分钟级/首,必须离线 + 并行 + 增量)。序列相似度是每次查询的剩余主成本,想再降一个数量级就缓存按库版本失效的 Music Space,或把 Levenshtein 换成 numpy/Rust 实现。
