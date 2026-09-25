# 能不能塞进手机(Python 之外的实现路径)

问题原样:"py 进不了手机,看能不能 cpp/rust"。这个文档用测量回答,不用形容词。
所有数字来自 93 首真实曲库(本地一个歌单文件夹,与 `docs/PERFORMANCE.md` 同一批),复现命令在文末。

## 结论
1. **在线打分路径可以完整搬到 Rust/C++,已证**。`rust/musicspace` 是一份只依赖 std 的实现,读导出载荷后重算 song→song 排名,与 Python 比对:**12 个 seed × 20 位次 = 240 个 slot,顺序 0 处不一致,分数差 0.0e0**。x86 release 实测 **6.1 毫秒/查询**,加载全库打分数据 **2.9 毫秒**。
2. **离线特征抽取不要搬手机**。单首 43.1 秒 CPU 里 `pyin` 28.7 秒 + `hpss` 5.9 秒(占 80%),93 首并行 6 worker 也要 22 分钟。这是 librosa/numba 的算法成本,换语言不会变快——变快要换算法(整段批处理 f0、CQT→更便宜的时频表示)或换硬件。
3. 所以手机形态是 **"桌面/服务器抽取 + 手机只做查询"**:抽取端产出 SQLite + embedding,同步到设备的是 **1560 字节/首 + 96 字节向量**,5000 首约 **8.3 MB**(音频本身除外)。

## 为什么载荷这么小
| 量 | 实测 |
|----|------|
| `music.db` 整行 | 11,231 字节/首(全列 8502 字节) |
| 打分真正读的列(`SPACE_COLUMNS`) | **1,560 字节/首**,其中三个序列列 586 |
| embedding | 24 维 × f32 = **96 字节/首** |
| FAISS 索引 + id 映射 | 328 字节/首 |

差距的来源:库里存了 `stat_vector`、`features_json`、段落/和声中间量,而查询只用分组特征 + 三条 token 序列 + 能量曲线。`SPACE_COLUMNS` 是由 `FEATURE_GROUPS` / `SEQUENCE_COLS` **推导**出来的常量,不是手写清单——加一个分组就自动进查询,不需要维护第二份列表。

## 一次查询到底做多少算术
| 量 | 值 |
|----|----|
| 重排候选对 | 92(93 首库;`faiss_top_k: 200` 封顶) |
| 分组相似度 | 92 × 9 分组 = 828/查询(采样口径 20 seed 合计 14,720) |
| Levenshtein 动态规划格 | 约 **1.28 M/查询**(20 seed 采样 25,573,526) |
| n-gram token | 约 26 k/查询 |

按 6.1 毫秒 / 1.28 M 格算,Rust 大约 **5 纳秒/格**(含 map 计数与分位查表)。手机 ARM 大核按桌面 x86 的 3–5 倍折损估:**单查询 20–30 毫秒**,一次滑歌手的推荐完全够用。曲库到 5000 首时把召回从 Flat 换成先做 f32 点积(5000 × 24 = 12 万次乘加,NEON 下亚毫秒)再重排 top-200,查询成本与库规模基本解耦。

## 已知边界(不要把这份证据用过头)
- **`rust/musicspace` 目前只覆盖 song→song**,没覆盖 `category` / `feed_next` / MMR 多样性。它们用的是同一批原语(分位、分组相似度、加权合并),移植是体力活而不是未知量,但**在完成前不能声称"三种模式都验过"**。
- **93 首小于 `faiss_top_k: 200`,所以两边候选集恒等**。曲库更大时 Python 只重排 FAISS 召回的 200 个,而端口若全库重排,结果可能不同(重排会改变顺序,库外的歌有概率进前 20)。真正上手机时必须把**两阶段召回一起移植**,否则这份等价性不覆盖那条路径。
- **embedding 后端是 PCA(ADR-3)**,端口只需 f32 点积;换成需要推理的神经 embedding 时,手机端要么带模型(几百 MB)要么退回"只同步已存向量"的形态——抽取端算好向量,设备只用不生成。
- 打分载荷是**私有曲库的指纹**。`data/portability/` 已加进 `.gitignore`:它只有内容哈希 id 和浮点特征,没有路径/标题,但仍然是"你听过什么"的信息,不该进版本库。

## 形态建议
- **Android**:`cargo ndk` 出 `cdylib`(armv7 / aarch64),Kotlin 侧 JNI 只暴露 `similar(track_id, limit) -> List<Hit>`、`category(...)`、`feed_next(...)`,SQLite 用只读副本。**不建议**把抽取放手机:pyin 那级成本在低端机会到几十秒/首。
- **iOS**:同一个 crate 编静态库,Swift 桥接。
- **C++ 替代路线**:如果目标形态已有 NDK/C++ 栈,把 `musicspace` 当**规范文档**用——它是这套打分语义唯一不依赖 Python 的可执行定义,照着翻成 C++ 也能保持等价(比对方式同样是 `--verify` 那段逐位次 diff)。
- **调参纪律**:`configs/weights.yaml` 改动后必须重新 `--export` 再跑 parity,否则比对的是旧语义。Rust 侧所有分组定义、权重、混合系数都从 `manifest.kv` 读,**代码里没有硬编码权重**——这是 ADR-7 在移植层的延伸。

## 复现
```bash
# 1. 导出打分载荷 + Python 自己的 top-20(约 1 分钟,含真实库 warm-up)
PYTHONPATH=src .venv/Scripts/python.exe scripts/bench_portability.py --json

# 2. 跑端口并验证等价(对照失败会退出 1)
cd rust/musicspace && cargo run --release -- data/portability --verify
```
输出四行:`loaded ... in 2.9 ms` / `median 6.09 ms/query` / `parity ... order mismatches = 0, max |score delta| = 0.00e0` / `control (one weight doubled): order mismatches = 183`。

**最后那行是这份证据成立的前提**。`--verify` 会把一个非 embedding 权重翻倍再比一次:如果那都抓不出差异,说明比较根本没执行,"0 不一致"就是假的。Python 侧的等价约束在 `tests/test_query_perf.py`——rapidfuzz 对教材版行 DP、`group_similarity` 对朴素参考实现逐组逐对比对,feed 的 MMR 和冷启动用最远点采样也一样验;`sequence_sim.py` 的换实现是以**排名逐位不变**为准出的(基线 worktree + 冻结时钟 + 同一份 DB 副本)。
