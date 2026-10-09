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
1. 查询侧(offline 抽取之外)的热点用**字节级等价**的方式优化,而不是换算法:编辑距离交给 `rapidfuzz`(C++,任意元素类型),`MusicSpace` 按库版本 `(COUNT, MAX(rowid), MAX(analyzed_at), MAX(meta_updated_at), MIN/MAX(track_id))` 做进程内缓存(见 ADR-16),`_score_pair` 每个分组只算一次,Feed 的 MMR 用一次矩阵乘代替逐对 numpy 标量调用。
2. 整条在线路径(分位归一化 → 序列相似度 → 加权合并 → song→song 重排)另写一份 **只依赖 std 的 Rust 实现** `rust/musicspace`,输入是 `scripts/bench_portability.py --export` 导出的打分载荷,输出与 Python 的 top-20 逐位次比对。
3. **离线抽取不做移植**(暂不搬 Rust/C++):`pyin` + `hpss` 占单首 43.1 秒里的 80%,那部分要换的是算法/模型(见 `docs/PORTING.md`),不是语言。
**理由**:手机上不了 Python,而"能不能塞进手机"必须用数字回答——载荷 1560 B/首 + 96 B 向量,单查询 92 对 × 9 分组 ≈ 1.28 M 编辑距离格,Rust x86 release 实测 6.1 ms/查询。等价比对是这一结论的前提:如果两份实现排名不同,"能移植"就没有意义。
**验证与代价**:`--verify` 里同时跑一个**反向对照**(把某个非 embedding 权重翻倍),对照必须报出差异,否则判为不可信并退出 1——纯 0 差异可能是比较根本没执行。实测:12 seed × 20 位次 = 240 slot,**顺序 0 处不一致、分数差 0.0e0**;对照抓到 183 处不一致。代价:Rust 端是打分载荷的**第二份实现**,分组定义/权重/混合系数必须从 `manifest.kv` 读而不是写死,`configs/weights.yaml` 改动后需重新 `--export`;导出物 `data/portability/` 是从私有曲库派生的特征指纹,不入库版本控制。

## ADR-16 Music Space 是库版本相关的派生数据,按版本指纹缓存
**决定**:`space.get_music_space(conn)` 用 `(sqlite 文件名, library_version)` 做 key 缓存全库分位;曲库变化(`library_version()` 的六元组 `(COUNT, MAX(rowid), MAX(analyzed_at), MAX(meta_updated_at), MIN/MAX(track_id))` 任一不同)即重建,且一次只驻留一个库(`_CACHE.clear()`)。每次查询只读打分真正用到的列(`SPACE_COLUMNS`,由 `FEATURE_GROUPS` + `SEQUENCE_COLS` + **词表额外列** `EXTRA_SPACE_COLUMNS` 推导,经 `PRAGMA table_info` 过滤)。
**理由**:分位与曲库强绑定(ADR-2),但重建全库分位实测只要 **6.5 ms**(占第一轮 0.55 秒的 1.3%),而原先**每次查询都重建一次**并全表 `SELECT *`(含 `stat_vector`、`features_json` 等大列)。真正的瓶颈是纯 Python Levenshtein:它占 0.504 秒采样里的 **0.452 秒**,且当时 `_score_pair` 把每个分组算了两次(一次进分数、一次进 detail),所以先去掉重复计算(→0.206 秒量级)再换 rapidfuzz,当前查询 21.4 ms。少读列几乎不为查询省时间,它的价值是让载荷估计(1560 B/首)与实际查询行为一致。
**代价**:长驻进程里同库并发写入时,缓存最多滞后一次 `library_version` 变化;`analyze` 写库后由版本号推进触发重建,不需要显式失效。

## ADR-17 类别来自共享词表 + 库内聚类自动发现,而不是只有人工预设
**决定**:类别路径改为三个来源共用一份事实。
1. `recommendation/lexicon.py` 是**唯一**的"维度 ↔ 查询字段名 ↔ 中文词"表(33 维、142 个词)。`category.DIM_MAP`、`discovery` 的命名、`text_query` 的解析全部从它派生,新增一个维度不需要在第二处登记;`EXTRA_SPACE_COLUMNS` 也由它推导,所以"词表里有、Music Space 读不到"这种静默失配会被 `test_every_preset_category_field_is_known` 当场抓住(本次就是这样抓到 `vocal_pitch_variance`、`mid_pitch_ratio`、`spectral_flux_mean`、`crest_factor`、`chorus_energy`、`chorus_repeat_count` 六维从未被读进空间,即任何以它们为目标的类别都在给一个不存在的维度排序)。
2. `recommendation/discovery.py`:在**分位空间**(不是原始单位)上做 KMeans,k 在 `[4,12]` 里按轮廓系数选,硬条件是每簇 ≥ `max(min_cluster_size, 4% × 库)`;簇名取质心偏离中位最多的 3 个维度的词表词。发现的类别查询目标**直接来自簇质心分位**,不再手调阈值,因此随曲库自动重标定。
3. `recommendation/text_query.py`:自由中文 → 词表查询。匹配不上的词(治愈、高级感)进 `unmatched` 并附原因,不映射到"差不多"的维度;否定词翻转目标分位。
`configs/categories.yaml` 6 → 22 条预设,`config.yaml: category_system` 一处调 `{engine, discovery, text, extra}`,`extra` 按 id 覆盖预设(用户本地新增不必改出厂文件)。
**理由**:用户反馈两点——"给出的维度其实都是我自己选的…估计后续会听腻",以及 93 首的小曲库下人工定的分位目标本就是猜的。库相对分位 + 聚类把"什么算一类"交回给曲库本身;词表则把"大家平时怎么描述歌"接进来(路线一),聚类命名是路线二。
**诚实边界**:真实 93 首库实测 k=4、轮廓系数 **0.106**,即簇之间边界本就模糊,4 个自动类别是"能过最小簇规模的全部",不是"曲库确实分成 4 类";`language`/`genre` 不参与聚类(它们不是音频特征),只作为簇名的 tag 后缀且要求覆盖率 ≥60%、纯度 ≥70%。名字只表示"这一簇在这几个可测维度上偏离中位",不是情绪或场景标签。
**代价**:聚类依赖 sklearn(可选导入,缺失时 `status: no-sklearn` 并退回纯预设);发现结果与分位同寿命,缓存在 Music Space 实例上,库版本一变就重算(93 首 34 维 k=4..12 全扫实测 551 / 696 毫秒,**每个库版本一次**,之后查询为 0 成本;见 `docs/PERFORMANCE.md` 第三轮)。

## ADR-18 语种 / 曲风只来自文件 tag,并参与缓存失效;无 tag 时退为"听感近似"且显式标注
**决定**:`tracks` 增 `genre` 与 `meta_updated_at` 两列。硬过滤 `language_is` / `genre_is` 只比对文件自带 tag(ADR-14 的延伸),库里无人有该 tag 时结果集为空并回报 `tag_missing` 计数——**不假装能听出语种**。没有 tag 可引时,曲风词退到 `GENRE_TERMS` 的**听感近似**(说唱 = 音域窄 + 节奏密 + 人声在),在 `matched` 里标 `side="proxy"`、词面自带"（按听感近似,非曲风标签）",绝不写进 `estimate_flags`。因为这两列现在被硬过滤和簇命名读取,`update_tags()` 只在值真的改变时 stamp `meta_updated_at`,`library_version()` 把它纳入指纹(ADR-16 六元组),tag 回填不可能留下一个陈旧 Music Space。
**理由**:第一版实现里 tag 写入不动 `analyzed_at`,回填语种后旧缓存仍会继续服务旧分位与旧簇名;而用户明确要"包括语言这些"的类别,若不做来源区分就会滑向规格 33 禁止的伪造。条件 stamp 而不是每次写都 stamp,是为了避免重复扫描同一批无 tag 的 wav 就把全库分位无谓重建。
**实测边界**:当前 93 首全是无 tag 的 wav,所以 `语种` 查询必然返回 `no-language-tag-in-library` / 空结果 + `tag_missing.language=93`,这是事实而非缺陷;`纯音乐` 则依赖 `has_vocal`(阈值化的估计值),因此 `filter_estimated` 会把它标出来——实测该过滤返回的 9 首里有 8 首从标题看是有唱的流行曲,即估计本身在这条上不可靠,标注只是不掩盖它,修正估计另列待办。
**代价**:两列 + 一次新旧值比较;`SPACE_COLUMNS` 多两列(仍是小标量,不影响载荷估计的打分列集合)。

## ADR-19 分析范围分两层:全曲只跑廉价帧级 DSP,昂贵特征仍锁在能量最密窗口
**决定**:`features/extract.py` 一次解码整曲拿到 `y_full` 后,`dynamics.frame_curves()` / `track_dynamics()` / `structure.structure_features()` 全部在**整段音频**上跑;`pyin`、和弦 CQT、HPSS 分解、人声/配器这些昂贵项继续只在 `_analysis_window()` 选出的窗口里算(默认 `audio.max_analysis_seconds: 45`,取滑动块能量 argmax 的连续段)。窗口起止写进 `features_json.analysis_window`,并新增 `analysis.analysis_scope: "full_track"` / `duration_seconds` / `feature_window_seconds`。`structure_features()` 现在**接收并忽略** `separated` 参数:HPSS 的 stem 来自那 45 秒,把它的水分贴到全曲段落上等于把错的数字安在对的时间;段落因此改报 `onset_activity`(打击活动代理量),不再报 `drum_energy`,也完全不报人声能量。
**理由**:改造说明 2.2 —— 旧 `energy_curve` 描述的是那 45 秒而不是这首歌,前奏、预副歌、桥段、尾奏系统性缺席,任何"高潮在哪"的判断都被母带最响的那一段绑住。全曲这一层刻意只用帧级廉价 DSP(定长 hop 的 RMS / 20–250 Hz 频带 / 色度 / onset envelope / 谱变化),不调 HPSS、不追基频,所以增加的是"整曲一遍分帧 STFT",不是"整曲一遍 pyin"。
**代价**:每首多跑一遍全曲 STFT(实测见 `docs/PERFORMANCE.md` 第四轮);`frame_curves` 的色度是全曲尺度,与 45 秒窗口的和弦估计从不在同一时间轴上,因此两者互不校验(结构层不读和弦序列)。

## ADR-20 边界 / 重复组 / 语义标签是三个结论,各有各的置信度;规则命名封顶 0.60
**决定**:`features/structure.py` 把三件事拆成三个字段并各自落列(`track_segments.boundary_confidence` / `group_label` / `label_confidence`),`label` 拿不准就是 `unknown`。
1. **边界**:`_novelty()` = 相邻块的色度距离(钳非负)与其余特征绝对差均值**相加**,再做块平滑;`_boundaries()` 用 `scipy.signal.find_peaks(distance=min_segment_seconds, prominence=0.20×(p95 − median))` 取**全部**突出点。超长段只在 novelty 高于中位的地方劈一次,劈不出证据就保持一段(不为显示上限伪造边界);头部或尾部残段短于 `min_segment_seconds` 一律被吸收。边界置信度是 novelty 在 p25–p98 区间的位置,钳在 [0.05, 0.99]。
2. **重复组**:`_assign_groups()` 在**三个家族**(chroma 0.60 / timbre 0.25 / density 0.15)上做加权余弦,每个家族**先减去本曲各段的均值**再 L2 归一(见 ADR-25;原来的"只归一、不减均值"在真实曲库上退化),再 average-linkage + `fcluster(criterion="distance")` 切,组名按首次出现编 A/B/C。**响度刻意不进相似度矩阵**——编曲加厚的副歌还是同一段落(改造说明 C.3–4 要求重复允许时间拉伸与编曲差异)。
3. **命名**:`_label_segments()` 只在"该段属于重复组"且"段自己的响度分位与密度分位都过门槛"时给 `chorus` / `verse`;`intro` / `outro` 另要求首(末)段且又静又稀。置信度**封顶 `label_max_confidence: 0.60`**,`pre_chorus` / `bridge` 干脆不输出——重复段规则分不清 bridge 与一段异常的 verse。
**理由**:旧实现按索引奇偶交替贴类型,并无条件把 `segment[0]` 叫 intro、`segment[-1]` 叫 outro,于是一首没有前奏的歌也会得到一串自信、内部自洽、全错的标签(改造说明 2.1)。检测到一处变化不等于这里是副歌,两段相似也不证明是同一个副歌;边界与命名分开写,是因为两者的改进路径完全不同(前者调门槛,后者要监督模型 = 阶段 D)。
**实测边界**:命名规则里"分组只提出候选,段自己的响度/密度定夺"来自一次真实回归——按分组平均功率判时,一段与副歌同和弦但很安静的 intro 会继承 `chorus`。分组相似度最初从"一个大拼接向量做 z-score"改成"三族各自归一",是因为只有几个段落时 z-score 会把噪声维上的微小差别变成一票否决;但**这一版在真实音频上仍然是坏的**(同一首歌里任意两段的原始余弦都贴着 1,分组全体退化),最终形态与实测数字见 ADR-25。
**代价**:界面必须把两个分数都摆出来才不构成误导(w-music 正在播放页每张段落卡片第三行就是「边界 x · 命名 x」);`unknown` 比例上升在验收里算**加分项**(改造说明 7.4)。

## ADR-21 和弦模板按根音平移(24 个互不相同的三和弦),并输出 margin
**决定**:`features/harmony.py` 原来的 `MAJOR` / `MINOR` 两个 12 维常量其实是**音阶轮廓**(`[1,0,1,0,1,1,…]`)却从不按根音平移,24 个模板里只有 2 个不同向量,所谓"根音比较"是在比同一个模板两次(改造说明 2.4)。现在改为 `triad_profile(root_pc, third)`:大三 = 0/4/7、小三 = 0/3/7,逐个根音平移;`templates()` 公开,并加 `template_check()` 自检 —— 实测 `n_distinct=24`、`min_pair_cosine=0.0`(C 与 C# 无共用音级)、`max_pair_cosine=0.6667`(C 与 Cm 共用根音与五度)。同时每个和弦在 `chord_confidences` 之外输出 `chord_margins`(冠军对亚军的余弦差)与 `chord_margin_mean`,折叠连续同名和弦时取 max。
**理由**:margin 接近 0 意味着这段色度和好几个三和弦一样贴合,即根音根本没定下来;没有这个字段,`chord_change_rate` 与置信度都无法区分"和声在动"和"匹配本身在抖"。`max_pair_cosine` 才是"模板塌回去了"的门(塌回旧写法时它是 1.0),而它现在的理论上限 2/3 也顺带说明三和弦模板天然模糊到什么程度。
**代价**:模板仍只认三和弦(七和弦/挂留不在此集合内),序列相似度继续走 ADR-4 的转调不变相对表示;`harmony` 分组权重保持不动——改造说明要求的是"修复之前别把旧根音序列当证据",本轮补的是可核查性,不是重估权重。

## ADR-22 `analysis_version` 是分析载荷的缓存键,并参与增量扫描判定
**决定**:`configs/config.yaml: analysis.version`(本节初值 2,现值 3——见文末与 ADR-25;全部曲线/边界/分组/命名门槛都在同一节下,`dynamics.DEFAULTS` 与 `structure.DEFAULTS` 镜像这些值以便独立调用),经 `utils/config.Config.analysis_version` 暴露。`tracks` 新增 `analysis_version` / `group_sequence` / `duration_ratios` / `dynamics_json` 四列,`track_segments` 新增 `group_label` / `label_confidence` / `boundary_confidence` 三列,都走既有 `PRAGMA table_info` + `ALTER TABLE` 补列(`_MIGRATIONS`,侧表另有 `_TABLE_MIGRATIONS`),旧库直接打开。`MusicLibrary.index(force=False)` 的跳过条件从"路径 + (size, mtime) 一致"升级为**还要求 `analysis_version` 一致**,不一致的文件进 `todo` 并在 summary 里报 `reanalysed_stale` 计数。`repository.get_analysis(conn, track_id, cfg.analysis_version)` 对旧版本行返回 `stale: True` 而不是 404,两个只读端点都带这个标记。
**理由**:算法换了而库里仍是旧载荷时,旧形状会一直冒充新形状。ADR-13 的 (size, mtime) 只回答"文件变没变",不回答"这份分析是不是当前算法做的";而版本不一致恰恰是唯一**该**重算的情形(改造说明 5.3)。回报 `stale` 而不是拒绝,是因为播放器画一条标明"旧版算法"的曲线比空白有用,前提是它明说。
**代价**:版本一 bump 就意味着整库重分析(93 首 6 worker 约 22 分钟量级,ADR-11/13),这必须是用户主动动作而不是播放器启动时的后台扫描。当前本机真实库 93 行 `analysis_version` 全为 NULL、`dynamics_json` 全为 NULL(`max(analyzed_at)` = 2026-09-25),即重新点「分析曲库」之前播放器只会显示"旧版分析,没有全曲响度曲线"这句说明。
**本 ADR 自己被抓到的一次兑现**:ADR-25 把重复分组从"原始家族余弦 + 门槛 0.90"改成"去均值后的家族余弦 + 门槛 0.20"——载荷列一个没变,变的是算出来的形状,正是这里说的"同一份旧载荷冒充新算法结果"。因此 `analysis.version` 已按本 ADR 的规矩 bump 到 **3**(`extract.py` / `utils/config.py` 的 fallback 同步),旧行的 `group_sequence` 会被判 stale 并计入 `reanalysed_stale`。

## ADR-23 结构相似度改三路可解释混合,不比 verse/chorus 字符串
**决定**:`recommendation/space.py` 的 `_structure_similarity()` 混合三个独立信号,权重 **0.45 / 0.30 / 0.25**:全曲能量轮廓的 Pearson(钳到 0–1)、A/B/C 重复组序列相似度(`SEQUENCE_COLS["structure"]` 由 `segment_type_sequence` 换成 `group_sequence`)、段落**时长占比**累计曲线重采样到固定 **8 点**后的 Pearson。缺哪一路就把权重摊到剩下的路上,而不是给 0 分。`GROUP_REASON["structure"]` 文案改为"能量轮廓/重复方式接近";`SPACE_COLUMNS` 补 `duration_ratios`(JSON 列,不由分组列与序列列推出,漏了就等于时长占比永远为空)。
**理由**:比较两首歌的 "chorus,verse,chorus" 字符串是在给两个猜测打一致性分,而猜测本身出自未封顶的规则(ADR-20,改造说明 5.1–5.2)。重复组序列、能量轮廓、段落时长占比每一个都能单独解释、也每一个都能由本地音频核查。4 段与 9 段的歌无法逐索引相关,所以时长占比取累计曲线并重采样到定点。
**代价**:`structure` 权重仍 0.05(改造说明 5.1 要求先降权),真实收益要等人工标注集;`scripts/bench_portability.py` 的导出列与 `manifest.kv` 必须同步——已把过期的 `structure_curve_blend: "0.6|0.4"` 换成 `structure_parts: "0.45|0.30|0.25"` + `duration_profile_points: "8"`,否则 Rust 端口在比一份已经不存在的定义。

## ADR-24 播放器只读一份分析载荷,且不为一条曲线冷启动引擎
**决定**:新增 `GET /v1/analysis?file_path=`(与 `GET /v1/tracks/{id}/analysis` 同一份载荷),一次带回 `duration_seconds` + `stale` + `dynamics.{time_seconds, rms_norm, rms_db, meta}` + `segments[].{start, end, label, group, boundary_confidence, label_confidence}`;w-music 侧 `RecommendService::ParseTimeline` 只按这些名字读,`Controls/StructureTimelineView` 是纯 C++ 的 Canvas 绘制,不复制任何 MIR 算法。x 轴取「`duration_seconds` / 曲线末格右边界 / 最后一个段落末尾」三者最大值。只有引擎已就绪或 `Settings().RecommendPrewarm()` 为真才发请求,否则界面直接说明原因。
**理由**:字段名是两端唯一的契约,改一个而不同步时界面上不会报错,只会是一张空白时间轴;所以 `tests/test_analysis_v2.py` 把键名和"曲线覆盖范围与段落末尾必须落在 `duration_seconds` 内"钉成断言,并用谎报时长(把 30 秒的歌报成 28 秒)做过反向对照——确凿失败在 `assert (29.75 + 0.25) <= (28.0 + 1e-06)`。`duration_seconds` 是容器声明的秒数,剪掉尾部静音或解码精度差异都会让它比真实音频短。冷启动 Python 要 5–10 秒,为一张从没人看的曲线付这个代价、并给没用过「个性推荐」的用户塞一个常驻进程,是不划算的。
**代价**:`dynamics_json` 是几千个浮点,只按单曲取、播放器进程内最多缓存 40 首并整份随「分析曲库」作废;曲线只画 `rms_norm`(曲内 0–1 相对尺度),绝对 dBFS 留在服务端不进界面——播放器没有跨歌比响度的入口,摆出来只会误导。

## ADR-25 重复分组必须减掉本曲各段的均值,阈值 0.20 是在真实曲库上数出来的
**决定**:`features/structure.py` 的家族相似度抽出为 `family_similarity(x, weights, center=True)`,返回 `(sim, used)`:`center=True` 时每个家族先减去**本曲各段在该维上的均值**再逐段 L2 归一;某个家族在本曲的段落之间完全不变(`max|u| < 1e-9`)就丢弃它,把权重摊给其余家族(`sim /= used`),三个家族都不变时每个段落自成一组。`center=False` 保留原样,只作为测试的对照臂存在。`group_similarity_threshold` 由 **0.90 改为 0.20**,`configs/config.yaml: analysis.version` 由 **2 改为 3**(`extract.py` / `utils/config.py` 的缺省值同步),于是旧行的 `group_sequence` 与由它推出来的 `chorus` / `verse` 标签都会被增量扫描判为陈旧。
**理由(实测,本机真实曲库 19 首,段落数 8–12)**:描述子的每一维都是正数(归一化色度能量、钳到 [0,1] 的音色/密度),同一首歌里**任意**两段的原始余弦都贴着 1 —— off-diagonal 中位 **0.954–0.994**,最低也有 0.845。按 0.90 切,**16/19 首整首歌并成一个组**(`AAAAAAAAAAAA`),页脚只能写「重复归为 1 组」;把 cut 推到 0.95 才勉强分出 2.11 组、推到 0.99 散到平均 5.89 组且还有一首全散开,也就是说原始余弦在任何阈值处都没有可用的平台。减去本曲均值后分布铺满整个区间(median −0.229…+0.106,max 0.718…0.996),阈值才真的有含义:共同模式(同调性、同一套乐器)本来就不携带"是否重复"的信息,段与段*偏离各自均值*的方向才是。
**阈值怎么定的**:在 −0.10…+0.625(步长 0.025)的网格上对 19 首逐一重切,看分组数与稳定度。**+0.20** 处每首平均 **3.63 组(2–6)**、最大组平均 5.47 段、**19/19 首都有 size≥3 的重复组**、0 首退化成一组也 0 首全散开,而且这一步只有 **1/19** 首的答案发生变化(全网格最稳的几格之一)。再往上到 +0.40 之后"最大组覆盖过半段落"从 10/19 掉到 3/19、+0.375 起每步 7 首改答案,同一段副歌被拆散,标签也就跟着丢重复证据。
**门与对照**:`tests/test_analysis_v2.py::test_repeat_grouping_needs_the_common_mode_removed` 用同一份构造数据跑两臂——对照臂断言原始余弦的成对相似度全挤在 0.05 宽的带里、且在 0.00–0.98 的**任何** cut 都切不出正确的两组划分,实验臂断言默认阈值切出 `ABABABAB`;把 `center` 改回 `False` 或把阈值改回 0.90,这条测试就会红。合成 fixture 那条(`test_structure_groups_repeats_without_inventing_labels`)当初是靠段落差异够粗才通过的,这正是它没拦住这个 bug 的原因——所以现在阈值与退化判定都以真实曲库为准。
**代价**:看到新分组必须重跑「分析曲库」(93 首 6 worker 约 22 分钟量级,ADR-11/13),本机库当前 93 行 `analysis_version` 仍全为 NULL、`max(analyzed_at)` = 2026-09-25。推荐侧无需改动:端口读的是存储的 `group_sequence` 字符串,分组算法变化不改变打分定义,但重算后的 `group_sequence` 会让结构分的第三路取值变化(ADR-23),`docs/PORTING.md` 那份等价性证据要在重分析后重跑一次 `--verify` 才算覆盖新数据。
