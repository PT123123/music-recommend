# CHANGELOG

本地音乐 MIR 特征提取与内容驱动推荐系统。按规格 Phase 1–7 逐竖切交付。

## 全曲结构与动态分析:修和弦模板 + 整曲曲线 + 重复分组(本次)
按 `docs/MUSIC_ANALYSIS_REFACTOR_PLAN.md` 的**阶段 A + B + C** 落地,并把结果接进播放器正在播放页的时间轴。阶段 D(歌曲功能标签模型)与阶段 E(乐器/人声/和声层)本轮**未做**,第 7 节的人工标注也未做。设计理由与代价记在 `docs/ADR.md` ADR-19 至 ADR-24。

- **修掉一个确定性的实现错误(ADR-21)**:`harmony.py` 里的 `MAJOR` / `MINOR` 两个 12 维常量是**音阶轮廓**(`[1,0,1,0,1,1,…]`)而不是三和弦模板,而且从没有按根音平移,于是 24 个"模板"实际只有 2 个不同向量——所谓 24 和弦匹配是在比同一个模板两次,根音比较没有意义(改造说明 2.4)。改为 `triad_profile(root_pc, third)` 逐根音平移(大三 0/4/7、小三 0/3/7),`template_check()` 自检实测 `n_distinct=24`、`min_pair_cosine=0.0`(C 与 C# 无共用音级)、`max_pair_cosine=0.6667`(C 与 Cm 共用根音与五度,即两个不同三和弦理论上能像到什么程度)。同时每步额外输出 `chord_margins`(冠军对亚军的余弦差)与 `chord_margin_mean`:margin≈0 表示这段色度和好几个三和弦一样贴合,根音并未被定下来——这是"能不能引用这条和弦序列"的核查字段。
- **新增 `features/dynamics.py`:全曲时间戳曲线 + 事件(ADR-19)**:整段音频按 0.5 秒定长桶聚合,输出 `rms_db`(dBFS,可跨曲比)、`rms_norm`(曲内 0–1)、`onset_density`、`low_band_energy`(20–250 Hz 占比)、`spectral_brightness`、`arrangement_change`(归一化谱流)六条同轴曲线,`meta` 里写清 `scope/time_semantics/rms_db_unit/interval_seconds/n_points/confidence_kind/calibrated`。事件 `build / peak / pull-back / drop / release` 从平滑曲线的相邻窗口对比产生(下降幅度与跨度决定是 drop、release 还是 pull-back),每个都带 `evidence` 与 `confidence`,且 peak 必须由第二个特征(onset / 低频 / 亮度)共同支持,`max_peaks` 上限 3;`min_dynamic_range_db: 1.5` 让一条被母带压平的曲目直接报告"没有动态可说"而不是编出事件。**响度地板 `rms_db_floor: -80 dBFS` 修掉一个假数字**:STFT 边缘帧此前能测到 -240 dB,把动态范围虚报成 240 dB。
- **重写 `features/structure.py`:边界 / A–B–C 重复组 / 诚实命名三分离(ADR-20)**:边界用 `find_peaks`(不是贪心 argmax——旧写法每轮都跳到 8–45 秒窗口内最强的那一处,一次很响的晚期转调会吞掉之前的所有边界,六段歌只回来两段),受 `min/max_segment_seconds` 与 `max_segments: 12` 约束,超长段**只在 novelty 高于中位处劈**,没有证据就保持一段;每边界带 `boundary_confidence`(novelty 在 p25–p98 的位置)。重复组在**三个家族**(chroma 0.60 / timbre 0.25 / density 0.15)上做加权余弦 + average linkage,`fcluster(criterion="distance")` 切;相似度要先减去本曲各段的均值、阈值 0.20,**这条初版写法在真实曲库上是退化的,修正见下面 ADR-25 那条**;**响度不进相似度矩阵**(编曲加厚的副歌还是同一段落)。命名封顶 `label_max_confidence: 0.60`,`pre_chorus` / `bridge` 不输出,其余留 `unknown`;分组只负责提出候选,段自己的响度与密度分位才决定它是不是副歌。旧实现按索引奇偶交替贴类型、无条件把首段叫 intro 末段叫 outro,这套逻辑已删除。
- **重复分组去掉了共同模式,阈值改成数出来的(ADR-25)**:家族余弦在归一前先**减去本曲各段在该维上的均值**,只保留"这一段偏离本曲平均值的方向";某个家族在本曲段落之间完全不变就丢弃、权重摊给其余家族,三个都不变时每个段落自成一组。`group_similarity_threshold` **0.90 → 0.20**,`analysis.version` **2 → 3**(旧行的 `group_sequence` 与靠重复证据给出的 `chorus` / `verse` 标签一律判陈旧)。
  - 为什么必须改:真实曲库 19 首(8–12 段)实测,原始余弦的成对相似度**中位 0.954–0.994、最低 0.845**——每一维都是正数,同一首歌里任意两段都像 0.95 以上,和"是否重复"无关。按 0.90 切,**16/19 首整首歌并成一个组**,正在播放页页脚只能写「重复归为 1 组」,改造说明 C.3「能指出重复片段」当时在真实音频上并不成立;cut 推到 0.95 才勉强 2.11 组、0.99 散到 5.89 组(还有一首全散开),原始余弦在任何阈值处都没有平台。
  - 阈值 0.20 的来源:在 −0.10…+0.625(步长 0.025)网格上对 19 首重切取最稳的一格——每首平均 **3.63 组(2–6)**、最大组平均 5.47 段、**19/19 首都含 size≥3 的重复组**、0 首退化成一组也 0 首全散开,且这一步只有 **1/19** 首改答案;再高到 +0.40 后"最大组覆盖过半段落"从 10/19 掉到 3/19,同一段副歌被拆散、标签跟着丢证据。去均值后分布铺满区间(median −0.229…+0.106,max 0.718…0.996),阈值才第一次有含义。
  - 门:`test_repeat_grouping_needs_the_common_mode_removed` 同一份构造数据跑两臂,对照臂(减均值前)断言"任何 0.00–0.98 的 cut 都切不出正确的两组",实验臂断言默认阈值切出 `ABABABAB`;把 `center` 改回 `False` 或阈值改回 0.90 都会红。合成 fixture 那条当初是靠段落差异够粗才通过的——这正是它没拦住这个 bug 的原因。
- **`structure_features()` 现在接收并忽略 `separated`**:HPSS 的 stem 来自那 45 秒窗口,把它的水分贴到全曲段落上是把错的数字安在对的时间上;段落改报 `onset_activity`,`drum_energy` 与人声能量都不再出现(改造说明 2.3:harmonic/percussive 分解不是源分离)。
- **`analysis_version` 成为分析载荷的缓存键(ADR-22)**:`configs/config.yaml` 新增 `analysis:` 节(当前 version **3** + 全部曲线/边界/分组/命名门槛),`utils/config.Config.analysis_version` 暴露;`tracks` 补 `analysis_version/group_sequence/duration_ratios/dynamics_json`,`track_segments` 补 `group_label/label_confidence/boundary_confidence`(旧库经 `PRAGMA table_info` + `ALTER TABLE` 直接打开)。增量扫描的跳过条件从"路径 + (size, mtime)"升级为**还要求版本一致**,不一致者计入新的 `reanalysed_stale`;`GET /v1/analysis` 对旧行返回 `stale: true` 而不是 404。
- **推荐侧不再比 verse/chorus 字符串(ADR-23)**:`_structure_similarity()` 改成三路混合——能量轮廓 Pearson **0.45** + A/B/C 重复组序列 **0.30** + 段落时长占比累计曲线重采样 8 点 **0.25**,缺一路就把权重摊到剩下的路上;`SEQUENCE_COLS["structure"]` 从 `segment_type_sequence` 换成 `group_sequence`,`GROUP_REASON["structure"]` 改为"能量轮廓/重复方式接近"(不再声称知道副歌结构),`SPACE_COLUMNS` 补 `duration_ratios`。`bench_portability.py` 同步导出列与 `manifest.kv`(旧键 `structure_curve_blend: 0.6|0.4` 换成 `structure_parts` + `duration_profile_points`),否则 Rust 端口在比一份已不存在的定义。
- **两个只读端点 + w-music 正在播放页时间轴(ADR-24)**:`GET /v1/tracks/{id}/analysis` 与 `GET /v1/analysis?file_path=`(播放器只有文件路径,内容哈希它算不出来)。播放器侧新增 `Controls/StructureTimelineView`(纯 C++ Canvas,96 柱位 + 14 色块池,同 `SpectrumView` 套路),段落色块按重复组取色、边界竖线透明度就是 `boundary_confidence`,每张段落卡片第三行直接摆「边界 x · 命名 x」;点时间轴或卡片即 seek。曲线只在引擎就绪或用户用过「个性推荐」页时请求,不为一张图付 5–10 秒冷启动。
- **两端都有机器门**:`tests/test_analysis_v2.py::test_analysis_endpoints_serve_the_timeline` 把播放器读的每个键名、曲线与段落必须落在 `duration_seconds` 之内、`time_semantics == "bucket_center"` 钉成断言;反向对照已实做——把时长谎报成 28 秒(真实 30 秒)后确凿失败在 `assert (29.75 + 0.25) <= (28.0 + 1e-06)`。客户端另有临时校验器把 `ParseTimeline` 里全部 `Find("…")` 键名与真实载荷逐一比对(缺失集为空)。
- **诚实边界**:本轮"分段是否合理"的证据来自合成 fixture(可控段落序列与响度包络)加少量人工听过的曲目,**没有标注集,因此不给准确率数字**;`analysis` 节里**分组阈值**已按真实曲库 19 首重标定(ADR-25),其余门槛(边界 prominence、命名分位、事件幅度)仍是合成数据的量级初值,换曲库后仍应按改造说明第 7 节重新标定。而且"分组数合理"是**分布意义上的**结论——阈值选在库层面最稳的一格,不等于每首都分对,单机听感核查仍未做。本机 93 首库仍是改造前分析的(`analysis_version` / `dynamics_json` 全为 NULL,`max(analyzed_at)` = 2026-09-25),播放器在此之上只显示"这首歌还是旧版分析,没有全曲响度曲线",直到用户点「分析曲库」。
- 测试:新增 `tests/test_analysis_v2.py`(11 项:模板两两不同 + 合成 C/F/G 听回根音 + margin、曲线覆盖整曲、事件能指出 build 与 peak、平直曲目不出事件、重复分组不伪造标签、重复分组必须先减本曲均值(含原始余弦对照臂)、节拍不足时退回时间网格、全曲载荷落库、版本变更才重分析、端点契约)。
- 回归:`.venv/Scripts/python.exe -m pytest tests -q` 全套 **64 项通过**(834.28 秒 / 13 分 54 秒,2026-10-08 夜间启动、跨午夜跑完,采集数与通过数同为 64)。分组改动只碰 `structure.py` 的内部算法与配置门槛,其余竖切(管线 / 扫描身份 / Feed / 查询等价 / 类别体系)未被带坏。

## 类别发散:共享词表 + 库内聚类自动发现 + 小曲库诚实标注(上次)
针对两条反馈——"给出的维度其实都是我自己选的,估计后续会听腻"和"曲库给得不多,小库里是否需要别的适配"——把类别从 6 条人工预设改成**三个来源共用一份词表**,并让自动类别的目标值来自曲库本身。新增 `docs/CATEGORIES.md`。

- **共享词表 `recommendation/lexicon.py`(ADR-17)**:33 个可测维度、142 个中文词,是"特征列 ↔ 查询字段名 ↔ 中文词"的唯一事实源;预设类别校验、聚类命名、自由文本解析全部从它派生。`EXTRA_SPACE_COLUMNS` 也由它推导——这条改动顺手抓到一个真实缺陷:`vocal_pitch_variance` / `mid_pitch_ratio` / `spectral_flux_mean` / `crest_factor` / `chorus_energy` / `chorus_repeat_count` **六维从未被读进 Music Space**,所以任何以它们为目标的类别一直在给一个根本没加载的维度排序(新增断言 `spec.col in SPACE_COLUMNS` 守住)。
- **聚类自动发现 `recommendation/discovery.py`**:分位空间 KMeans,k 按轮廓系数在 `[4,12]` 选,硬门槛每簇 ≥ `max(4, 4%×库)`;簇名取质心偏离中位最多的 3 个维度。**查询目标直接来自簇质心,不再手调分位**,因此曲库换了类别定义自动重标定。真实 93 首实测 `k=4 / 轮廓系数 0.106 / 4 簇(38·23·17·15)`,文档里明确写出"边界模糊,4 类是门槛内的全部结果,不是曲库确实分 4 类"。库太小则拒绝输出(`status: library-too-small`),不会造出单曲"类别"。
- **自由中文文本 `recommendation/text_query.py` + `--text` / `text` 接口**:最长匹配、否定翻转目标(`不要快节奏` → 目标 0.15)、量不出来的词(治愈/高级感)进 `unmatched` 并附原因,不映射到相近维度。词表补齐了用户自己的说法:`节奏感强`、`重低音`、`低音重`、`念白`、`女生/男声`(见下)。
- **小曲库诚实适配**:`support/candidate_pool/low_support/min_support`(低于 `max(5, 5%)` 标记)、`filters_applied` + `filter_estimated`、`unknown_filters`(写错的条件只报告不生效)、`tag_missing` 计数、`scored_dims/unscored_dims/filter_only`(过滤后没维度可排时如实说明,不再给全池 1.000 的假排名)。结果近重复压制改为贪心 `λ·相关 − (1−λ)·最大余弦`(默认 λ=0.8、余弦 >0.97 丢弃),没有 embedding 时不做并如实报 `deduped: False`。
- **语种 / 曲风只读文件 tag(ADR-18)**:`tracks` 增 `genre` + `meta_updated_at`,`language_is`/`genre_is` 精确匹配,无 tag 时结果为空并回报缺失计数——**不假装能听出语种**。无 tag 可依时曲风退到 `GENRE_TERMS` 听感近似、`女声/男声` 退到 `VOCAL_PROXY_TERMS` 音区近似(性别没有任何本地特征能识别),两者在 `matched` 里标 `side="proxy"` 并在 CLI 明写"按…近似"。因为这两列现在被硬过滤和簇命名读取,`update_tags()` 仅在值真的变化时 stamp,`library_version()` 升级为**六元组**,tag 回填不可能留下陈旧 Music Space;重复写入相同 tag 不再让缓存失效(测试 `test_rescanning_unchanged_tags_does_not_invalidate_the_space` 守住)。
- **修掉 `instrumental` 硬过滤缺陷**:旧实现把 `has_vocal` 的条件当"字段存在"处理,`纯音乐` 实际从未过滤任何人;现在 `HARD_FILTERS` 是谓词表,`_passes_hard` 支持谓词 / tag 相等 / 通用 `<col>_min|_max`,并新增反向对照(要人声必须给出补集)。诚实边界同时写明:当前 93 首里判为无人声的 9 首按文件名看 8 首是演唱曲,即该估计实测不可靠,所以它出现在 `filter_estimated` 里而不是被当成事实——修正估计本身另列待办。
- **预设 6 → 22 条**(`configs/categories.yaml`),`config.yaml` 新增 `category_system: {engine, discovery, text, extra}` 一处调参,`extra` 按 id 覆盖出厂预设(用户加类别不必改出厂文件);新增 `scripts/categories.py`(列表 + `--refresh` + `--json`)。
- 测试:新增 `tests/test_category_system.py` 23 项(合成 3 族 × 10 首、全词表列都填值,所以"预设类别必须可回答"是真断言而不是空跑)。按项目惯例每项都带反向对照,例如"未知过滤条件必须报告且不删任何人"、"没有 embedding 时 `deduped` 必须是 False 而不是谎称做过"。
- **去重写第一版就被自己测出来两个问题**(见 `docs/PERFORMANCE.md` 第三轮):(1) Python 三重循环让 category 从 3.8 毫秒变成 **27 / 36 毫秒**,即"去重"本身就是那 7 倍退化——改成一次堆矩阵 + 每选一首一次矩阵向量乘,回到 5.5 毫秒;(2) 存的 embedding **不是单位长度**,`v @ c` 是内积不是余弦,配置里 0.97 这个阈值比的其实是模长,现在先 L2 归一,并由 `test_dedupe_similarity_is_an_angle_not_a_magnitude`(同方向、0.4 模长的重复必须被压掉)钉住。`deduped` 的语义同时收紧为"本轮确实因冗余丢了候选",另有 `duplicates_suppressed` 给条数。
- 依赖:聚类走 `sklearn`(已装),惰性导入,缺失时 `status: no-sklearn` 并退回纯预设。

## 在线打分路径 25 倍 + 移动端可行性验证
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
