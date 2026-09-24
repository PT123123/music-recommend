# Performance Notes

## 已知瓶颈:pyin 基频 + hpss 分解(CPU,离线)
`features/melody.py`(pyin f0)与 `features/extract.py`(hpss)是整条流水线主要开销。

## 真实曲库实测(本地歌单文件夹,93 首 WAV)
93 首 WAV / 3.7 GB,时长 76–433 秒(均值 231 秒)。

| 配置 | 结果 |
|------|------|
| 整曲分析(`max_analysis_seconds: 0`),单进程 | 单首 297 秒曲目 = **172 秒**抽特征 → 93 首约 **4 小时** |
| 45 秒分析窗 + 6 worker | 93 首全部入库 = **约 32 分钟**(≈20.6 秒/首有效),0 失败 |
| 加载后查询(song→song / category / feed) | **亚秒级**;93 向量 FAISS 检索毫秒级 |

全库入库耗时拆解:`normalize`(解码整曲)≈13 秒/首,`extract`(45 秒窗)在 6 worker 争抢下 ≈108 秒 → 并行摊薄到 ≈20 秒/首。机器 22 核,worker 数仍可上调,`normalize` 的整曲解码是剩余固定成本。

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
| 重排 | O(K) | 每对约 8 分组,分组内 O(seq_len²) 序列距离 |

## 后续可优化方向
1. **再加并行**:22 核机器上 `--workers` 仍可上调(12–16);剩余固定成本是 `normalize` 解码整曲,可改成 ffmpeg `-t/-ss` 只解码目标窗口,把 ≈13 秒/首也省掉。
2. **缓存 raw analysis 中间量**:把 chroma / hpss / f0 结果落盘(比 wav 小),重排调参时秒级重跑。
3. **换后端 embedding 时一次性重算**:改 `configs/config.yaml: embedding.backend` → 重跑 `build_embeddings` + `build_faiss`,SQL 主表不变。
4. **测试加速**:临时库 fixture 复用同一份特征缓存,只重建 index。
5. **磁盘**:`normalize` 会为每首写出 44.1kHz 单声道 WAV,93 首占用 **1.8 GB**(`data/normalized_audio/`)。该目录是可重建缓存,删掉后重扫即可恢复。
