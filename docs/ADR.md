# Architecture Decision Records

关键设计决策与理由。对应规格章节号。

## ADR-1 特征全部来自本地音频,禁止伪造(规格 核心原则 / 23 / 33)
**决定**:凡本地无法可靠得到的属性一律留空或标 `unknown`,并在 `estimate_flags` 中标注为估计值。
- `language` → 不猜(留空),禁止从文件名伪装成音频分析结果。
- `vocal_gender` → 恒 `unknown`,只作辅助过滤,不作核心距离(规格 13)。
- `fine_genre_tags` / 命名字器概率 → 不产出(轻量方案无可靠模型)。
- "抖音热门"类外部流行度 → 不提供;`local_hot` 仅用本地行为统计(规格 33, 66)。
**理由**:规格核心原则要求内容由音频推导。伪造字段会污染 Music Space 与推荐解释。

## ADR-2 Music Space 用全库经验分位归一化(规格 14 / 60)
**决定**:每个标量特征在**当前本地库内**转成经验 CDF → [0,1],再按语义分组。
**理由**:实现「这首歌在本地库处于什么位置」的相对度量(如"全库高音最突出的一批"),且天然消除量纲差异(规格 60)。代价:分位依赖曲库,换库需重算(可 `rebuild_embeddings` 重建)。

## ADR-3 embedding 可替换,SQLite 为准 FAISS 为缓存(规格 24 / 58)
**决定**:`AudioEmbedder` 抽象接口;当前实现为 librosa 统计向量 → StandardScaler → PCA(whiten)。FAISS 索引随时可由 `track_embeddings` 重建。
**理由**:MVP 不下载大模型(PANNs/MERT),接口保留替换点。SQLite 是唯一真相源,FAISS 丢失可 rebuild。

## ADR-4 和弦/旋律/段落序列做转调不变比较(规格 10 / 11 / 19)
**决定**:和弦用 `{(root-tonic)%12}:{quality}` token;旋律用相对音高/音程序列;相似度 = Levenshtein+n-gram+Jaccard+transition 加权(0.35/0.25/0.2/0.2),不只用 Jaccard。
**理由**:降低移调影响;多指标混合比单一 Jaccard 更稳。

## ADR-5 类别与 Feed 共用同一 Recommendation Core(规格 42)
**决定**:唯一区别是查询向量来源——类别来自 `categories.yaml` 的目标分位,Feed 来自 `InterestModel` 的兴趣向量;召回与重排同一套。
**理由**:避免两套推荐系统漂移。

## ADR-6 Feed 防坍缩:preference+similarity+novelty+diversity(规格 39 / 41 / 65)
**决定**:MMR 贪心做多样性,近期历史窗硬排除重复,冷启动用最远点采样铺开后探索。权重全部配置化(`config.yaml: feed`)。
**理由**:纯相似串联会坍缩到极小区域;需邻近探索而非全同。

## ADR-7 权重/阈值/类别全部配置驱动(规格 62 / 63)
**决定**:`weights.yaml`(分组权重 + active 开关)、`config.yaml`(feed 参数)、`categories.yaml`(类别 → Music Space 查询)。核心算法不硬编码数值。
**理由**:分析真实曲库后可直接调参,不改代码。

## ADR-8 长曲目用"能量最密窗口"做有界分析,而非整曲或前奏
**决定**:`audio.max_analysis_seconds` 配置一个连续窗口,`_analysis_window()` 用滑动块能量 argmax 选出全曲最响的一段来抽特征;`duration` 列仍写整曲真实时长,窗口起止单独存入 `features_json`。
**理由**:真实曲库单首 4 分钟时整曲抽特征要 172 秒(93 首约 4 小时),而取开头 N 秒会落在前奏/主歌,使 `chorus_*`、`energy_curve`、人声维度系统性失真。取最响窗口是对"用户记忆集中在副歌"(规格 17)的直接呼应。属于有损近似,因此段落/和声/旋律等本就已在 `estimate_flags` 中标为估计值。

## ADR-9 刚 like 过的歌计入"近期历史"并从 Feed 排除(规格 65)
**决定**:`recent_track_ids()` 的判定事件包含 `like`。
**理由**:like 只可能发生在已听过的歌上,一秒后把它再推一遍正是规格 65 要防的"刚播放完 A 又推荐 A"。`dislike` 不进此列表——它的排斥由负 `base_weight` 负责,两套机制不该混用。
**副作用记录**:这暴露出 `test_feed_biases_to_liked_family` 原断言(`fams.count("high") >= 2`)实际只在缺陷行为下成立——被喜欢的那几首本身就是 high 家族。已改为同库双用户的差分断言。

## ADR-10 序列相似度有 token 预算,旋律按 2 半音分桶(修 266 秒)
**决定**:`features.max_sequence_tokens`(默认 120)对所有参与 Levenshtein 的序列做均匀降采样;旋律序列先中值滤波(`melody_pitch_smooth_frames: 5`)再按 `melody_note_quantise_semitones: 2` 合并成"音级事件",`interval_histogram` 仍在**全量**事件上统计,不受预算影响。
**理由**:真实曲库里逐帧旋律序列长 2677–4015 token,Levenshtein 是 O(n·m) 纯 Python,song→song 对 93 首库实测 **266 秒**。降到 120 token 后实测约 1 秒。
**为什么是 2 个半音**:pyin 逐帧误差本身就有 0.3–0.5 半音,与一个半音同量级。真实音频上量得:1 半音分桶时 83–90% 的"音高变化"只是跨取整边界的抖动翻转;2 半音 + 5 帧中值把它降到 0,而 `pitch_range` 只差 ≤1 半音(30/34/32 vs 30/34/33)。代价是**小于 2 半音的级进会被合并**——这是承认听音分辨率的边界,而不是假装能分辨;按核心原则,与其伪造精度不如降级合并。

## ADR-11 pyin / hpss 每首只算一次,结果注入各特征模块
**决定**:`features/extract.py` 统一算一次 `hpss` 与一次 `pyin`(在谐波分量上),把 `separated` / `f0` 作为可选参数注入 `melody` / `vocal` / `instruments` / `structure`;模块单独调用时仍自行计算(参数默认 None)。
**理由**:原先 pyin 跑 2 遍、hpss 跑 3 遍。人声模块复用同一条 f0 后只需按人声频带(80–1100 Hz)筛掉框外帧。实测 `pyin` 占单首抽取 43.1 秒里的 28.7 秒,去重直接决定墙钟时间。

## ADR-12 track_id 由文件内容(大小 + 首尾块)哈希,而非路径
**决定**:`preprocess/audio.track_id_for()` 读 `size` 与首/尾各 256 KB 做 SHA-1 前 16 位;文件不可读时退回路径哈希。`tracks.file_path` 仍 UNIQUE,内容相同、路径变化时更新该列而不是新增行。
**理由**:路径哈希下用户改歌单目录名或重命名文件,会让 `interactions` / `evaluations` 里指向旧 id 的历史全部变孤儿(规格 49/56 的行为数据是推荐的核心输入)。重编码文件得到新 id 是正确的——音频内容变了,相似性也确实变了。
**代价**:同内容出现在两个目录时只保留一行(后扫描者更新 `file_path`),即"按内容去重";`data/normalized_audio` 的缓存文件名随 id 方案改变而全量失效,需要重解码一次。

## ADR-13 重扫按 (size, mtime) 跳过未变更文件
**决定**:`tracks.file_size` / `file_mtime` 落库,`MusicLibrary.index(force=False)` 用 `repository.scan_state()` 比对后只分析变化/新增文件;`--force` / `ScanRequest.force` 用于特征版本变更时全量重算。跳过的文件仍会补读一次 tag(`repository.update_tags()` 只写 tag 列,不动 `stat_vector` / `analyzed_at`)。
**理由**:单首全解码+抽特征 30–40 秒,重扫一遍 93 首是 40 分钟量级的纯浪费;跳过判定只需 stat。判定条件要求 `stat_vector IS NOT NULL`,因此中途崩溃留下的半行会被重算,不会永久跳过。

## ADR-14 文件自带 tag 是唯一允许进入的外部信息,并显式标注来源
**决定**:`preprocess/metadata.read_tags()` 用 mutagen 读 title/artist/album/year/language,`tracks.meta_source` 记 `'tag'` 或 `'none'`;读不到就留 NULL——**不从文件名猜**歌名歌手,也不把 tag 内容混进 `estimate_flags`。HTTP 结果里的 `display` 块与 CLI 标签都带上来源可追溯性。
**理由**:规格 4/81 禁止"用外部平台标签当推荐依据"和"为空字段编造值",但禁止的是**伪称音频分析得出**;文件自己携带的文本是事实,只要来源标注清楚就不是伪造。播放器集成也确实需要这些字段(否则列表只能显示文件名)。
**实测边界**:用户的 `~/Music` 里 mp3 有 40/40 带 tag,而本次入库的 93 首 **wav 全部无 tag**(`meta_source='none'`),所以该曲库的标题列仍为空——这正是标注来源的意义:缺就是缺。

## ADR-15 在线打分路径先做等价优化,再以 std-only Rust 参考实现验证可移植性
**决定**:
1. 查询侧(offline 抽取之外)的热点用**字节级等价**的方式优化,而不是换算法:编辑距离交给 `rapidfuzz`(C++,任意元素类型),`MusicSpace` 按库版本 `(COUNT, MAX(rowid), MAX(analyzed_at), MIN/MAX(track_id))` 做进程内缓存(见 ADR-16),`_score_pair` 每个分组只算一次,Feed 的 MMR 用一次矩阵乘代替逐对 numpy 标量调用。
2. 整条在线路径(分位归一化 → 序列相似度 → 加权合并 → song→song 重排)另写一份 **只依赖 std 的 Rust 实现** `rust/musicspace`,输入是 `scripts/bench_portability.py --export` 导出的打分载荷,输出与 Python 的 top-20 逐位次比对。
3. **离线抽取不做移植**(暂不搬 Rust/C++):`pyin` + `hpss` 占单首 43.1 秒里的 80%,那部分要换的是算法/模型(见 `docs/PORTING.md`),不是语言。
**理由**:手机上不了 Python,而"能不能塞进手机"必须用数字回答——载荷 1560 B/首 + 96 B 向量,单查询 92 对 × 9 分组 ≈ 1.28 M 编辑距离格,Rust x86 release 实测 6.1 ms/查询。等价比对是这一结论的前提:如果两份实现排名不同,"能移植"就没有意义。
**验证与代价**:`--verify` 里同时跑一个**反向对照**(把某个非 embedding 权重翻倍),对照必须报出差异,否则判为不可信并退出 1——纯 0 差异可能是比较根本没执行。实测:12 seed × 20 位次 = 240 slot,**顺序 0 处不一致、分数差 0.0e0**;对照抓到 183 处不一致。代价:Rust 端是打分载荷的**第二份实现**,分组定义/权重/混合系数必须从 `manifest.kv` 读而不是写死,`configs/weights.yaml` 改动后需重新 `--export`;导出物 `data/portability/` 是从私有曲库派生的特征指纹,不入库版本控制。

## ADR-16 Music Space 是库版本相关的派生数据,按版本指纹缓存
**决定**:`space.get_music_space(conn)` 用 `(sqlite 文件名, library_version)` 做 key 缓存全库分位;曲库变化(`library_version()` 的五元组任一不同)即重建,且一次只驻留一个库(`_CACHE.clear()`)。每次查询只读打分真正用到的列(`SPACE_COLUMNS`,由 `FEATURE_GROUPS` + `SEQUENCE_COLS` 推导,经 `PRAGMA table_info` 过滤)。
**理由**:分位与曲库强绑定(ADR-2),但重建全库分位实测只要 **6.5 ms**(占第一轮 0.55 秒的 1.3%),而原先**每次查询都重建一次**并全表 `SELECT *`(含 `stat_vector`、`features_json` 等大列)。真正的瓶颈是纯 Python Levenshtein:它占 0.504 秒采样里的 **0.452 秒**,且当时 `_score_pair` 把每个分组算了两次(一次进分数、一次进 detail),所以先去掉重复计算(→0.206 秒量级)再换 rapidfuzz,当前查询 21.4 ms。少读列几乎不为查询省时间,它的价值是让载荷估计(1560 B/首)与实际查询行为一致。
**代价**:长驻进程里同库并发写入时,缓存最多滞后一次 `library_version` 变化;`analyze` 写库后由版本号推进触发重建,不需要显式失效。
