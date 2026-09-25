# CHANGELOG

本地音乐 MIR 特征提取与内容驱动推荐系统。按规格 Phase 1–7 逐竖切交付。

## 在线打分路径 25 倍 + 移动端可行性验证(本次)
把"能不能塞进手机"变成有数字的答案:先证明查询侧还能再快一个数量级,再把整条查询路径重写一份只依赖 std 的 Rust 实现,用逐位次比对证明两边等价。

- **song→song 0.55 秒 → 21.8 毫秒**(ADR-15、ADR-16):
  - `recommendation/sequence_sim.py` 的编辑距离交给 `rapidfuzz`(C++,任意 token 类型)。消融很干净:在当前代码上把它换回纯 Python 行 DP,单次查询 **21.4 → 227.7 毫秒**,即这一项占剩余成本的 91%。旧实现还把每个分组算两遍(一次进分数、一次进 detail),`core.py` 改为单次计算后 DP 单元数减半——两者相加才是 0.452 秒 → 0.206 秒的关系。
  - `recommendation/space.py` 新增 `get_music_space()` 进程内缓存,key 为 `(sqlite 文件名, library_version)`,`repository.library_version()` 用五元组 `(COUNT, MAX(rowid), MAX(analyzed_at), MIN/MAX(track_id))` 做版本指纹;曲库变即重建,一次只驻留一个库。顺带修正了旧结论:**全库分位重建只要 6.5 毫秒**,它从来不是那 0.55 秒的来源。
  - 查询改为只读打分列:`SPACE_COLUMNS` 由 `FEATURE_GROUPS` + `SEQUENCE_COLS` 推导(加一个分组自动进查询,无需维护第二份清单),经 `PRAGMA table_info` 过滤后走 `repository.rows_with_columns()`。
- **feed_next 0.099 → 3.3 毫秒、category 0.026 → 3.8 毫秒**(ADR-15):`interest/feed.py` 的 MMR 从"每对候选 × 每首已选调一次 `np.linalg.norm`"(93 首库 20 项批次约 15 万次 numpy 标量调用)改成一次矩阵乘 + 布尔掩码索引;`_cold_start()` 用最远点采样的向量化版本,质心取 `argmin(‖M − centroid‖)`。
- **修掉一个向量化引入的静默语义缺陷**:`dmax` 以 0 初值 `np.maximum` 累积 `1 - cos`,而候选与已选歌**反相关时 `1 - cos` 会大于 1**,钳在 1.0 就压平了整批多样性分数——A/B 里表现为 feed 第 1 步赢家改变(旧实现该位置 div = 1.4637)。改为 `have_sel` 标志 + 精确逐对公式。广播形状错的另两类会抛异常当场暴露,这一类只改语义、静默通过。
- **等价性优先于速度**:基线 worktree(上一个提交)+ 冻结 `time.time` + 同一份真实 DB 副本,逐位次 A/B 比对排名与分数;`rapidfuzz` 对教材版行 DP、`group_similarity` 对朴素参考实现逐组逐对(>1000 次比较)断言 `<1e-12`,feed MMR 与冷启动同法验证 → 新增 `tests/test_query_perf.py`(7 项)。其中 feed 的 fixture 刻意把 embedding 摆在圆周上并断言 `best_div > 1.0`,否则它根本抓不到上面那个钳位缺陷。
- **可移植性实测**:`scripts/bench_portability.py` 回答三件事——载荷(打分列 **1560 字节/首** + embedding 96 字节,整行 11,231)、算术(92 对 × 9 分组,约 **1.28 M** 编辑距离格/查询)、等价(`--export` 写出无 JSON 的 `tracks.tsv` + `manifest.kv` + `expected_top20.tsv`)。5000 首约 **8.3 MB** 打分载荷,音频除外。
- **`rust/musicspace`(只依赖 std 的端口)**:分位(`partition_point`)、interned-token Levenshtein、n-gram/Jaccard/transition 计数、能量曲线 Pearson 混合、分组加权重排、f32 累加以对齐 FAISS 余弦。`--verify` 实测 **12 seed × 20 位次 = 240 slot,顺序 0 处不一致、分数差 0.0e0**,`median 6.09 ms/query`、加载 2.9 毫秒。**同一命令内置反向对照**:把一个非 embedding 权重翻倍后必须抓出差异(实测 183 处),否则退出 1 —— "0 不一致"若是比了个空就毫无意义。
- **诚实边界**(写进 `docs/PORTING.md`):端口目前只覆盖 song→song,`category` / `feed_next` 尚未移植;93 首 < `faiss_top_k: 200` 使两边候选集恒等,库变大后必须连两阶段召回一起移植,否则这份等价性不覆盖那条路径;**离线抽取不要搬手机**(单首 43.1 秒里 pyin 28.7 + hpss 5.9,是 librosa 的算法成本),手机形态是"抽取在桌面/服务器、设备只做查询"。
- 依赖:`requirements.txt` / `pyproject.toml` 增加 `rapidfuzz`。`.gitignore` 增加 `data/portability/`(私有曲库的特征指纹,只有内容哈希与浮点,但仍属"你听过什么")与 `rust/**/target/`。
- 文档:新增 `docs/PORTING.md`;`docs/ADR.md` 增 ADR-15(等价优化 + 端口 + 反向对照)、ADR-16(Music Space 按库版本缓存);`docs/PERFORMANCE.md` **第二次勘误**——旧文把 0.55 秒归因于"重建分位 + 纯 Python Levenshtein",实测分位只占 1.3%,归错方向会去优化一个 1% 的东西。

## 真实查询延迟修正 + 扫描/身份/元数据四项
在真实曲库上把"查询亚秒级"这句未经核实的声明拆开验证,发现并修掉四项,同时补上文件自带元数据的读取。

- **song→song 266 秒 → ~0.55 秒**(ADR-10):旋律序列原本逐帧存,真实曲目 2677–4015 token,Levenshtein 是 O(n·m) 纯 Python,20 候选实测 **266 秒**。新增 `features.max_sequence_tokens`(120)均匀降采样预算 + `melody._note_events()` 先把 pyin 轨迹转成音级事件;`interval_histogram` 仍按全量事件统计,不被预算截断。93 首真实库最终实测 10 次均值 **0.55 秒**(最大 0.65 秒),category 0.026 秒、feed 0.099 秒。
- **旋律分桶 1 → 2 个半音**(ADR-10):真实音频量得 1 半音分桶时 83–90% 的"音高变化"只是取整边界抖动;配 `melody_pitch_smooth_frames: 5` 中值滤波后该比例归零,`pitch_range` 保持在 ≤1 半音误差内。代价明确记录:小于 2 半音的级进会被合并——不假装拥有 tracker 没有的分辨率。
- **抽取去重**(ADR-11):`extract.py` 统一算一次 `hpss` 与一次 `pyin` 并注入 `melody` / `vocal` / `instruments` / `structure`(原来是 pyin×2、hpss×3);实测单首 43.1 秒里 pyin 28.7 秒、hpss 5.9 秒。93 首 6 worker 全库重扫 32 分钟 → **22 分钟**。测试 `test_pyin_and_hpss_run_once_per_track` 用 monkeypatch 计数守住这条。
- **增量扫描**(ADR-13):`tracks.file_size` / `file_mtime` 入库,`MusicLibrary.index(force=False)` 跳过未变更文件,`scan_library.py --force` / `POST /v1/library/scan {force}` 用于特征版本变更;被跳过的文件仍补读一次 tag(`update_tags()` 只写 tag 列,不动 `stat_vector` / `analyzed_at`)。实测第二次扫描 0.0 秒。`schema.py` 加 `_migrate()`(`PRAGMA table_info` + `ALTER TABLE`),旧库不改结构即可升级,已在 93 行旧库副本上验证四个新列补齐、行数不变。
- **track_id 改内容哈希**(ADR-12):路径哈希 → `size + 首尾 256KB` 的 SHA-1 前 16 位,重命名/挪目录不再把 `interactions` / `evaluations` 变成孤儿;`test_track_id_survives_rename_and_copy` 守住这条。旧真实库(93 行,路径 id)已挪入 `.quarantine-20260925/music-old-pathhash-trackid.db`,重新全量入库。
- **读取文件自带 tag**(ADR-14):新增 `preprocess/metadata.py`(mutagen,惰性导入,读不到不报错),`tracks` 增 `year` / `meta_source` 列并写入 title/artist/album/year;`repository.display_map()` 一次查询补齐,`/v1/recommend/*` 与 `/v1/feed/next` 结果新增 `display` 块(含 `meta_source`),CLI 标签优先用 tag、退回文件名。诚实边界:tag 是文件自己的声明,永不进入 `estimate_flags`,也不从文件名猜;实测该 93 首 **wav 全部无 tag**(`meta_source='none'`,标题列仍为空),而 `~/Music` 下 40 首 mp3 有 40 首带 tag——同一份代码,两种事实。
- 依赖:`requirements.txt` / `pyproject.toml` 增加 `mutagen`。
- 测试:新增 `tests/test_scan_identity.py`(11 项:身份稳定、tag 读/无 tag、tag 落库、`update_tags` 不触发重算、增量跳过四情形、半行不算已分析、pyin/hpss 计数、序列预算、抖动抑制)。两条失败先查自己的断言:±0.4 半音抖动不跨取整边界(改用 60.5±0.35 的真实抖动模型)、`update_tags` 测试传了自相矛盾的 `meta_source`,均为测试写错而非实现错。
- 文档:`docs/PERFORMANCE.md` 更正"加载后查询亚秒级"这条未经真实曲库核实的旧声明,并补入上表实测数字。

## 真实曲库首跑 + 可用性修正
- 首次跑真实曲库(本地一个歌单文件夹,93 首 WAV / 3.7 GB):全部入库,0 失败;非音频文件(`.txt`/`.url`)按扩展名正确跳过。
- `config.yaml: audio.max_analysis_seconds`(默认 45):`extract._analysis_window()` 取**能量最密集的连续窗口**(滑动块能量 argmax),而非开头,保证覆盖副歌;`duration` 仍为整曲真实时长,窗口起止记入 `features_json`。单首 172 秒 → 并行摊薄到约 20 秒。
- `MusicLibrary.index(workers=N)` + `scan_library.py --workers` + `POST /v1/library/scan {workers}`:抽特征走 `ProcessPoolExecutor`,并把 BLAS/numba 线程收敛到 1;DB 写入仍集中在主进程,`TrackFeatures` 跨进程回传。
- **Feed 重复排除修正**:`behavior.recent_track_ids()` 之前只认 `play/complete/replay`,导致刚 `like` 的歌会立刻被再次推流(违反规格 65)。`like` 现已计入近期历史;不喜欢的排斥仍由负权重负责。
- `tests/test_feed.py::test_feed_biases_to_liked_family` 重写为**差分断言**:同一库上两个用户分别喜欢 high / low 家族,各自 top1 必须落在被喜欢的家族。原断言其实只在"重新推出刚喜欢的那首"时才成立,掩盖了上面的缺陷;且合成 fixture 的同频克隆会被 MMR 多样性项正确压制,计数型断言无法表达意图。
- `tests/verify_mvp.py` 增加守卫:该控制校验靠文件名 `<family>_<n>.wav` 判定家族,对真实曲库无意义,现改为检测到低下划线比例时打印 SKIP 而非给出自信的错数字。
- 清理 `api/server.py` 模块文档字符串中"category/feed 尚未实现(501)"的过期说明。

## Phase 7 — 人工评测导出
- `database/behavior.py`: `record_evaluation` / `export_evaluations_csv`(`seed,recommendation,label`)/ `evaluation_stats`。
- `scripts/evaluate.py`: 交互式打分、记录、导出 CSV、统计。
- `POST /v1/evaluate` 端点。

## Phase 6 — HTTP API 补全 + CLI
- `api/server.py`: 全部端点落地(health / tracks / categories / recommend/similar / recommend/category / feed/next / feed/feedback / feed/state / feed/reset / evaluate / library/scan / tracks/index)。
- 与 Python API 共用同一 `Recommendation Core`(规格 53)。
- `scripts/run_server.py`、`scripts/recommend.py --category`、`scripts/feed.py`。

## Phase 5 — 动态 Feed / 无脑流
- `interest/model.py`: `InterestModel`,指数衰减 `base_weight * exp(-age/tau)`,short/medium/long 三窗口按 `λ1>λ2>λ3` 融合(规格 35–38)。
- `interest/feed.py`: `FeedEngine`,候选召回 + preference/similarity/novelty 打分 + MMR 多样性 + 近期重复排除 + 冷启动最远点采样(规格 39–41, 65–66)。
- 冷启动使用本地行为统计多样性,不冒充外部热度(规格 66)。

## Phase 4 — 固定类别推荐
- `recommendation/category.py`: `CategoryEngine`,Hard Filter + Soft Ranking(规格 64),`resolve()` 区分可用/忽略维度。
- `configs/categories.yaml`: 仅用本地可推导维度(energy / calm / percussive / bright timbre / high female vocal / deep low vocal)。
- 类别从 YAML 读取,不硬编码(规格 63)。

## Phase 3 — 结构 / 分段
- `features/structure.py`: 能量分层 + agglomerative 分段,`segment_list` / `segment_type_sequence` / `energy_curve`(64 bin)/ `chorus_*`。
- `recommendation/space.py`: `_structure_similarity()`(energy-curve 相关 + 段序列),`structure` 分组进入重排。
- `track_segments` 独立建表(规格 55)。

## Phase 2 — 和声 / 旋律 / 人声 / 配器 + 重排
- `features/harmony.py`: chroma 模板匹配 24 三和弦,转调不变 `{offset}:{quality}` 相对序列(规格 10)。
- `features/melody.py`: pyin f0,pitch 统计 + low/mid/high ratio + interval 直方图 + 相对音高序列(规格 11–12)。
- `features/vocal.py`: 启发式人声维度;`vocal_gender` 恒为 `unknown`(规格 13–14)。
- `features/instruments.py`: bass/drum/低中高能比;命名字器概率故意不产出。
- `recommendation/sequence_sim.py`: Levenshtein + n-gram + Jaccard + transition 加权(规格 10)。
- `recommendation/core.py`: 多分组加权重排 + 逐对权重归一化 + 基于真实计算的 `reasons`(规格 28, 59–61)。

## Phase 1 — MVP 端到端
- 预处理(44100 mono)、全局声学/节奏特征、librosa→PCA embedding、FAISS、Music Space 分位归一化、song→song、`library/scan`、`tracks/index`、合成测试音频、`verify_mvp` 控制校验。
