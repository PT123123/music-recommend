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
| song→song 查询(预算 120 token + 2 半音分桶) | 10 次平均 **0.55 秒**(最大 0.65 秒) |
| category 查询 / feed_next | **0.026 秒** / **0.099 秒** |
| **song→song 查询(rapidfuzz + Music Space 缓存 + 分组单次计算,当前)** | **0.0218 秒**(5 次平均) |
| **category 查询 / feed_next(当前)** | **0.0038 秒** / **0.0033 秒** |

**勘误**:此行原写"加载后查询亚秒级",那是合成语料(4 秒短片、序列仅几十 token)上的结论,真实曲库实测 266 秒。已按真实数据改正,并记下修复位置(`features.max_sequence_tokens`)。
**第二次勘误**:本文件曾把第一轮之后剩余的 0.55 秒归因于"每次调用重建 Music Space 分位 + 纯 Python Levenshtein"。实测全库分位重建只要 **6.5 ms(1.3%)**,几乎不是瓶颈;真正的开销是纯 Python Levenshtein(0.504 秒采样里 **0.452 秒**)。归因错误会把优化引向错误的方向(先去缓存分位收益不到 2%),所以记下:定位瓶颈要靠计时,不能靠"看起来复杂的那部分"。

## 第二轮:在线查询路径 0.55 秒 → 0.022 秒(25 倍),全程要求排名逐位不变
三项改动都必须**字节级等价**,否则"变快了"没有意义(等价性由 `tests/test_query_perf.py` 的朴素参考实现断言)。

| 改动 | 位置 | 收益(实测) |
|------|------|------|
| 编辑距离交给 rapidfuzz(C++),对任意 token 类型 | `recommendation/sequence_sim.py` | 现在把 rapidfuzz 换回纯 Python DP,song→song 从 **21.4 → 227.7 毫秒**,即单这一项就占当前查询的 91%。与上表 0.452 秒的旧测量自洽:当时 `_score_pair` 把每个分组算两遍(DP 单元数×2 → 约 412 毫秒) |
| 每对的分组相似度只算一次(`detail` 复用) | `recommendation/core.py` | 分组相似度调用 2×→1×;与上一行合并构成 0.55 秒到 0.022 秒的主要部分 |
| Music Space 按库版本缓存 + 每次只读打分列 | `recommendation/space.py`, `database/repository.py` | 全库分位重建 **6.5 毫秒**;真正省下的是原先每查询都 `SELECT *`(含 `stat_vector`/`features_json` 大列) |
| Feed 的 MMR 改成一次矩阵乘 + 布尔掩码索引 | `interest/feed.py` | feed_next **0.099 → 0.0033 秒**。原先每个已选 × 每个候选调一次 `np.linalg.norm`,93 首库 20 项批次里约 15 万次 numpy 标量调用——开销全在调用不在算术 |

当前真实曲库(93 首)实测:`similar 21.8 ms` / `category 3.8 ms` / `feed_next 3.3 ms`,façade 冷建 18 → 8 ms。

**向量化过程中修掉的真实缺陷(必须记录)**:`dmax` 初值为 0 并用 `np.maximum` 累积 `1 - cos`,而**候选与已选歌反相关时 `1 - cos` 可以大于 1**,把上限钳在 1.0 会把整批多样性分数压平——A/B 里表现为 feed 第 1 步赢家改变(旧实现该位置 div = 1.4637)。修法是 `have_sel` 标志 + 精确逐对公式,不做任何截断。教训:等价的"形状改写"要用逐位 A/B 验证,向量化的广播错误里有两类会抛异常(当场发现),这一类只改语义、静默通过。

## 可移植性实测(手机能不能装下)
`scripts/bench_portability.py` 把问题拆成"载荷 / 算术 / 等价"三块,结论详见 `docs/PORTING.md`。

| 量 | 实测 |
|----|------|
| 打分所需列 | **1560 字节/首** + embedding 96 字节(24 维 f32) → 5000 首约 **8.3 MB**(不含音频) |
| 整行 / FAISS | 11,231 字节/首(全列 8502)/ 328 字节/首 |
| 单次 song→song | 92 对候选 × 9 分组 = 14720 分组相似度 / 20 seed 采样 **25,573,526** 编辑距离格(约 1.28 M/查询) |
| Rust(std-only,`rust/musicspace`) | 93 首全库 **6.1 毫秒/查询**(x86 release),加载 2.9 毫秒 |
| Python ↔ Rust 排名 | 12 seed × 20 位次 = 240 slot:**0 处顺序不一致、分数差 0.0e0**;反向对照(权重翻倍)抓到 183 处 |

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
1. **在线路径已经没有动机再优化**:93 首库 21.8 毫秒,再往下是常数级微调。真正剩下的两个数量级来源在**离线抽取**(单首 43.1 秒里 pyin 28.7 + hpss 5.9)和**候选规模**——曲库到 5000+ 首时 `faiss_top_k: 200` 仍是 200 对重排,但 FAISS 从 Flat 换 IVF/HNSW 才谈得上规模;移动端方案见 `docs/PORTING.md`。
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
| song→song 单次 | **21.8 毫秒**(第二轮;第一轮 0.55–0.65 秒) |
| 打分真正需要的列 | **1560 字节/首** + embedding 96 字节 —— 其余 8.5 KB/首是抽取中间量,查询根本不读(`SPACE_COLUMNS`) |

结论:元数据体积小到一个数量级——整库常驻内存毫无压力,进程起来 4 毫秒就能读全表,所以**查询层不需要为"少取数据"设计**;真正需要处理的是抽特征阶段(分钟级/首,必须离线 + 并行 + 增量)。序列相似度这个剩余主成本已由 rapidfuzz 消掉,在线路径进入 20 毫秒量级,因此可以直接搬到手机端(见 `docs/PORTING.md`)。
