# 完全去 Python:现状、runtime 边界与三条路线

问题原样:"看下能不能完全剥离 py 用 cpp/rust" + "我不想有 runtime 负担,python runtime
负担太大,还有别的方案吗"。本文把两问合起来回答:先量清 Python 现在到底还剩在哪,
再按"要不要连桌面抽取端的 Python 也消掉"给三条路线。测量口径与 `PORTING.md` 同一批
93 首真实曲库。

## 现状:Python 还剩在哪

| 侧 | 规模 | 覆盖 | 未覆盖 |
|----|------|------|--------|
| `rust/musicspace` | 528 行,std-only | song→song 在线打分;与 Python **240 slot 排名逐位一致**(顺序 0 处不一致、分数差 0),6.1 ms/查询,加载全库 2.9 ms | category / feed_next / MMR / 两阶段召回 / 离线抽取 |
| Python(`src/` + `scripts/`) | 约 3850 行 | 全部:MIR 抽取(librosa)、PCA、FAISS、SQLite、FastAPI、Feed、CLI | — |

两处等价性边界(`PORTING.md`「已知边界」原样有效):

1. 端口没覆盖 category / feed / MMR——它们用的原语(分位、分组相似度、加权合并)
   端口里已有,移植是体力活不是未知量,但完成前不能声称"三种模式都验过"。
2. 93 首 < `faiss_top_k: 200`,两边候选集恒等,两阶段召回路径还没被等价性覆盖;
   曲库变大后必须把召回一起移植。

## 关键澄清:runtime 负担已经被隔离在一个时刻

Python 只出现在**一个时刻:把一首没分析过的新歌加进库**。查询路径(设备上真正跑的
东西)已经是 Rust,0 Python——这是 `rust/musicspace --verify` 证明过的事实。

按形态拆开:

- **手机/App 端**:SQLite + 1560 字节/首打分载荷 + 96 字节向量 + Rust `cdylib` →
  **Python runtime 负担 = 0**。见 `PORTING.md` 的 Android/iOS 落地清单。
- **曲库预建、随包分发**:连桌面也永远碰不到 Python。
- **桌面端**:只有"加新歌"需要 Python,且是一次性批处理,不是常驻 runtime。

所以"设备上没有 Python"**已经达成**。真正待决策的只剩一件事:桌面抽取端那 3850 行
Python 要不要也消掉、怎么消。

## 完全剥离的可行性分层

**① 在线路径补齐(可控,体力活)** — category / feed / MMR / 两阶段召回用的都是端口
里已有的原语,照 `musicspace` 这份"唯一不依赖 Python 的可执行规范"翻即可;再加一个
Rust HTTP 层(如 axum)替代 FastAPI、SQLite 只读读取。做完后**运行态全链路 0 Python**,
且每一步都可以沿用 `--verify` 式的逐位次 parity 验收。

**② 离线抽取(大工程,且是决策点)** — 相当于重写一个 librosa 子集:

- `pyin`(f0)一项占 28.7 秒/首(全曲 43.1 秒的 67%),是概率 YIN + numba 实现,
  直接端口成本最高;
- HPSS、节拍、和弦、结构分段都是 librosa 算法,每个都要移植 + parity 验收;
- 关键事实:43 秒/首是 **librosa/numba 的算法成本,不是 Python 的语言成本**,换语言
  不会变快——变快只能换算法(整段批处理 f0、更便宜的时频表示)或换硬件;
- 一旦换 f0/CQT 方案,特征值会变 → **全库重抽、PCA/FAISS 重建、现有 240 slot 逐位
  等价基线作废重建**。这不是坏事(aubio 级 f0 会把抽取提速一个量级),但必须当作
  一次带基线重建的算法变更来走,见下文「换算法的纪律」。

## 三条路线(消掉桌面抽取端的 Python)

| 方案 | 做法 | Python 残留 | 代价 | 基线影响 |
|------|------|------------|------|----------|
| **A. 冻结成单文件 exe** | PyInstaller 把 `extract_features.py` 打包成 `extractor.exe`,双击跑 | 无(Python 变成构建期产物) | 最小,一两天,算法零改动 | 无 |
| **B. 抽取端 Rust 化** | symphonia 解码 + aubio(C 静态链)做 f0 + rustfft 做 MFCC/chroma | 无 | 大;重写 librosa 子集 + parity 逐项验收 | 换 f0 → 全库重抽 + 基线重建 |
| **C. 把 Python 瘦到骨头** | 砍掉 librosa/numba/faiss/sklearn:便宜 f0 + 手写 MFCC/chroma + 暴力点积代替 FAISS | 只剩 numpy+scipy+soundfile,freeze 后也很小 | 中 | 同 B(换了 f0) |

两个顺带事实,三条路线都受益:

- **FAISS 本来就可以扔**:5000 首 × 24 维 f32,暴力点积在 NEON 上亚毫秒
  (`PORTING.md` 已算过这笔账),它只是可重建缓存。任何方案里都可以顺手去掉这个依赖。
- **pyin 换 aubio/YINFFT 反而是好事**:它占 28.7 秒/首(全曲成本 67%),换了之后
  抽取速度提升一个量级——A 以外的方案都顺手解决这个老问题(见 `PERFORMANCE.md`)。

## 推荐

- 目标是"手机上没有 Python" → **已达成,不用动**(`PORTING.md` 接 JNI/静态库即可)。
- 目标是"自己/用户机器上也不装 venv" → **A**,80/20:一个 `extractor.exe` + 一个
  Rust 查询二进制,Python 彻底变成构建工具。
- 目标是"整个仓库源码级无 Python" → **B**,工程最大但终态最干净,顺手把 43 秒/首
  的抽取干到秒级。C 是 A 与 B 的折中:改动集中在特征层,依赖树最小,但同样要重建基线。

## 换算法的纪律(选 B/C 时生效)

换 f0 实现等于换特征定义,必须当一次正式算法变更走:

1. 先跑一次旧库全量导出,冻结旧基线(`bench_portability.py --export` + 冻结 DB 副本);
2. 新 f0 上线后全库 `--force` 重抽、重拟合 PCA、重建 FAISS;
3. `tests/test_query_perf.py` 的等价约束改为对新基线成立;
4. 在 `ADR.md` 记一条:换掉了什么、为什么、新旧特征不可比、库版本如何区分。

## 决策记录

- 2026-09-25:分析完成,**路线未选定**。当前交付形态 = Rust 查询端(已验证)+ 桌面
  Python 抽取端(现状),即 `PORTING.md` 的"桌面抽取、设备只查询"。选定路线后把
  决定补进 `ADR.md`,本文的表格与基线纪律随之收敛为已选项的执行清单。
