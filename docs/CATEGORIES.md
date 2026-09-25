# 类别推荐说明(预设 + 自动发现 + 自由文本)

一句话:**类别不是标签,是当前曲库里的一块区域。**所有目标值都是"库内分位",所以同一句"高能量"在 93 首和 3000 首里指的不是同一批歌(ADR-2)。

三种入口都在同一条打分路径上,返回的诚实元数据也相同。

## 1. 三种入口

```bash
# (a) 预设类别 id
PYTHONPATH=src python scripts/recommend.py --category deep_low_vocal

# (b) 自动发现的类别(auto-* 只能随曲库存在,列表里带成员数)
PYTHONPATH=src python scripts/categories.py            # 列表 + 聚类状态
PYTHONPATH=src python scripts/categories.py --refresh   # 重跑聚类,不吃缓存
PYTHONPATH=src python scripts/recommend.py --category auto-02

# (c) 自由中文(路线一:大家平时怎么描述歌)
PYTHONPATH=src python scripts/recommend.py --text "来点安静又明亮的纯音乐"
PYTHONPATH=src python scripts/recommend.py --text "高音女声"
PYTHONPATH=src python scripts/recommend.py --text "不要快节奏的"
```

HTTP:`POST /v1/recommend/category`,body 里 `category_id` / `text` / `query`(+`hard_filter`)三选一;
`GET /v1/categories?include_discovered=1`、`GET /v1/categories/discovery?refresh=1`。

## 2. 出厂预设 22 条(`configs/categories.yaml`)

每条只写"这个项目真的从音频里量出来的维度"。想加自己的类别:在 `configs/config.yaml` 的
`category_system.extra` 里按 id 追加或覆盖,**不必改出厂文件**。

| id | 名称 | 依据(软排序维度 / 硬过滤) |
| --- | --- | --- |
| `high_energy` | 高能量 | 响度·节奏密度·鼓能量 |
| `calm_low_stimulation` | 低刺激 / 平静 | 响度低·节奏稀疏·偏暗 |
| `percussive_rhythmic` | 强节奏 | 节奏密度·可舞性·速度 |
| `bright_timbre` | 明亮音色 | 音色明暗 |
| `high_female_vocal` | 极致高音女声 | 人声音区·音域跨度·演唱强度 / 有人声 |
| `deep_low_vocal` | 低沉人声 | 人声音区低·响度中 / 有人声 |
| `instrumental` | 纯音乐 / 无人声 | 人声占比极低 / `instrumental` |
| `slow_ballad` | 慢速抒情 | 速度慢·响度中·偏暗 / `bpm ≤ 85` |
| `fast_intense` | 快速激烈 | 速度快·节奏密·响度高 |
| `rap_like_flow` | 说唱感(音域窄 + 节奏密,估计值) | 人声音域窄·节奏密·人声在 / 有人声 |
| `whisper_soft` | 轻声耳语 | 演唱强度低·响度低·动态小 / 有人声 |
| `vocal_power` | 人声爆发 | 演唱强度高·音域宽·响度高 / 有人声 |
| `deep_bass_heavy` | 低频厚重 | 低频能量比·低音成分·偏暗 |
| `bright_airy` | 通透明亮 | 音色明暗·音色对比·响度中 |
| `dark_tense` | 暗沉紧张 | 音色暗·不协和度高 |
| `wide_dynamics` | 大动态起伏 | 动态范围·能量波动 |
| `steady_groove` | 稳定律动 | 节拍稳定·可舞性·速度波动小 |
| `harmonic_loop` | 和声简单循环 | 和声变化率低·和声节奏慢 |
| `wide_melody` | 旋律大跨度 | 旋律跨度·旋律起伏 |
| `breathy_texture` | 气声 / 沙哑质感 | 噪声成分比·过零率 |
| `chorus_lift` | 副歌拉满 | 副歌能量·响度 |
| `speechy_vocal` | 念白感人声 | 人声音高起伏小·人声占比高 / 有人声 |

`high_female_vocal` 的名字沿用规格里的原始类别名,但**没有任何本地特征能识别性别**——它排的是人声音区/
音域,`note` 字段就是为这件事写的,`scripts/categories.py` 会跟着名字一起打出来。

## 3. 自动发现(`auto-*`)

`recommendation/discovery.py`:在**分位空间**上做 KMeans,k 在 `[4,12]` 之间按轮廓系数选,硬门槛是每簇
`≥ max(min_cluster_size, min_cluster_share × 库大小)`;簇名取"质心偏离中位最多"的前 3 个维度的词表词
(`lexicon.name_from_deviations`)。

关键区别:**发现的类别查询目标直接来自簇质心分位**,不是人调出来的数。曲库换了,类别和它的目标值一起重标定。

真实曲库(93 首 WAV,34 维)实测:

```
曲库 93 首 | 聚类状态 ok | k=4 | 轮廓系数=0.1057 | 特征维度=34
  auto-03  人声突出·低音轻·整体音区高        成员=38
  auto-02  旋律跨度大·音域跨度大·旋律起伏大    成员=23
  auto-04  圆润·暗沉·低频成分厚              成员=17
  auto-01  律动舞曲感·低频成分厚·节奏密集      成员=15
```

轮廓系数 0.106 要如实读:**簇与簇之间边界模糊**,4 个类别是"能过最小簇规模门槛的全部结果",不是"曲库确实
分成 4 类"。k=4 是门槛内选出的最高分,不是置信度。

全无人声的簇会改名成"无人声·XXX"并自带 `hard_filter: {instrumental: true}`;只有当某个 tag 在簇内
覆盖 ≥60% 且纯度 ≥70% 时才允许在名字后面写"(tag:语种 X)",没有 tag 就一个都不写。

## 4. 自由文本能听懂什么

词表在 `recommendation/lexicon.py`,33 个可测维度、142 个中文词(高/低两侧),是**唯一**的
"维度 ↔ 查询字段 ↔ 中文词"来源:预设类别、聚类命名、文本解析共用它。新增维度只需改这一处
(`EXTRA_SPACE_COLUMNS` 由它推导,所以不会出现"词表里有、Music Space 读不到"——这条有测试守着,
本次就是它抓出 6 个维度从未被读进空间)。

- 否定会翻转目标:`不要快节奏` → 速度分位目标从 0.85 变 0.15。
- 最长匹配优先:`低沉人声` 不会被拆成 `人声`。
- 语种/曲风走 tag(`language_is` / `genre_is` 精确匹配文件自带 tag)。
- 情绪/场景词(治愈、伤感、高级感)**不在词表里**,因为没有任何本地测量能支撑它们——它们会进
  `unmatched` 并附原因,而不是被映射到"差不多"的维度上。

## 5. 每次回答都带的诚实元数据

| 字段 | 含义 |
| --- | --- |
| `used_dims` / `ignored` | 真正参与排序的维度;查询里本地量不出来、被丢掉的字段 |
| `support` / `candidate_pool` / `low_support` / `min_support` | 有多少首真的落在目标附近 / 池子多大;低于 `max(5, 5%)` 即标记。**小曲库最要紧的一个数** |
| `filters_applied` | 实际生效的硬过滤(值为假的条件不算生效) |
| `filter_estimated` | 其中哪些依据的是阈值化的估计值(目前只有 `has_vocal` / `instrumental`) |
| `tag_missing` | 按 tag 过滤时,库里有多少首根本没这个 tag |
| `unknown_filters` | 写错了的过滤条件名——只报告,不静默生效 |
| `matched` / `unmatched` / `cleaned` | 文本解析:命中词(含维度与方向)、答不了的词与原因、去掉已理解部分后剩下的原话 |
| `genre_proxies` / `vocal_proxies` | 没有 tag 可依时的"近似":曲风按听感近似、女声/男声按人声音区近似,`matched` 里标 `side="proxy"` |
| `scored_dims` / `unscored_dims` / `filter_only` | 过滤之后还剩哪些维度真的有数据;一个维度都没得排时结果是纯过滤列表,不是排名 |
| `deduped` / `duplicates_suppressed` | 本轮是否因冗余丢了候选、丢了几条(比较的是归一化后的夹角;没有 embedding 时不做,并如实报 `False`) |

结果去重:贪心 `λ·相关 − (1−λ)·与已选的最大余弦`(默认 `λ=0.8`,余弦 >0.97 直接丢)。93 首库上
一个类别的前 30 名原本几乎全是同一族歌的重复,分数本身看不出冗余,只有向量能看出来。
存的 embedding **不是单位长度**,所以比较前先 L2 归一:否则"最大余弦"比的其实是模长,一首音量小的重复
能躲在 0.97 之下(这一点由 `test_dedupe_similarity_is_an_angle_not_a_magnitude` 守住)。

## 6. 已知边界(不要外传成"能识别")

1. `纯音乐` 依赖 `has_vocal = 人声频段能量比 > 0.25 且周期度 > 0.2`。当前 93 首里被判为无人声的 9 首,
   按文件名看有 8 首是流行演唱曲——即这条**实测不可靠**,所以它进 `filter_estimated`。修正估计本身是另一条待办。
2. 语种:该库 93 首 wav **全部无 tag**,`英文/中文` 查询必然返回 `no-language-tag-in-library` 且结果为空。
3. 曲风:同一原因,当前只能走听感近似(`genre_proxies`),名字里带"说唱感…估计值"就是这个意思。
4. 聚类依赖 sklearn;缺失时 `status: no-sklearn`,列表退回只有预设。
5. 参数在 `configs/config.yaml` 的 `category_system` 一处:`engine`(支持度门槛、去重)、
   `discovery`(k 范围、最小簇、命名阈值、tag 门槛)、`text`(高/低目标分位、同维度是否保留首个词)。
