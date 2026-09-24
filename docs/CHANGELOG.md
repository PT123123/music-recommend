# CHANGELOG

本地音乐 MIR 特征提取与内容驱动推荐系统。按规格 Phase 1–7 逐竖切交付。

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
