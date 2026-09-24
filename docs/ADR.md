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
