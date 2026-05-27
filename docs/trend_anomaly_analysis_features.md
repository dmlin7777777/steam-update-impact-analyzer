# 趋势与异常分析 — 特征说明文档

本文档说明 `trend_anomaly_analysis.py` 脚本中生成、使用或依赖的特征，包含每个特征的计算方法、数据契约、边界情况与示例。目标读者为数据工程师与特征工程师。

## 输入数据契约

脚本期望的输入 DataFrame（通常为 FE 输出或原始评论数据）需至少包含如下列：

- `review_date`：评论的时间戳（字符串或 pandas 可解析的日期格式），用于生成时间序列特征。必须可转换为 pandas datetime。
- `vader_compound`：情感评分（数值，范围通常在 -1 到 1），用于情感统计与趋势计算。
- `review_content_clean`：用于协同行为检测的文本（TF-IDF 相似度）。

可选列（若存在会被用于更多特征或检测）：

- `engagement_score`：交互/参与度分数（如点赞/回复综合指标）。
- `toxicity_score`：毒性或违规评分。
- `votes_up` / `text_length`：用于互动异常检测与归一化。
- `dominant_topic`：主题标签（用于稀有主题检测）。
- `appid`：应用/游戏 ID，用于按应用分组提取时序特征。

输入约束与预处理：

- `review_date` 建议按升序排序（脚本会在提取处进行排序处理）。
- 缺失数值列会在计算时被 fillna(0) 或在统计中忽略（视具体特征而定）。

## 输出契约

特征提取器返回一个 DataFrame，包含原始列 + 新增的趋势/异常特征。名称遵循可读前缀便于后续使用（如模型训练或告警）。

异常检测器会在 DataFrame 上新增若干布尔/数值列来标记异常。


## 生成的特征清单与计算方法

下面按类别列出脚本中生成的特征、计算公式与实现细节。

### 1) 时间相关基础特征

- `review_datetime`：由 `pd.to_datetime(review_date)` 得到的 datetime 对象。
- `hour`：`review_datetime.dt.hour`，0-23 整数。
- `day_of_week`：`review_datetime.dt.dayofweek`，0=周一。
- `is_weekend`：当 `day_of_week` 为 5 或 6 时为 1，否则 0。

用途：捕捉日内/周内周期性。



### 2) 滚动窗口统计（基于时间窗，默认 windows = [24,48,72] 小时）

对于每个窗口 w（例如 24），脚本会基于时间偏移（DatetimeIndex）生成以下特征（列名前缀包含窗口，窗口字符串格式如 `'24H'`）：

- `sentiment_rolling_mean_{w}h`：基于时间窗口（例如 `rolling('24H')`）计算当前时间点向前 `w` 小时范围内的情感均值。

- `sentiment_rolling_std_{w}h`：情感的时间窗口滚动标准差，空值在后续预处理或模型输入阶段会被稳健插补。

- `sentiment_rolling_min_{w}h` / `sentiment_rolling_max_{w}h`：时间窗口内的最小/最大情感值。

- `neg_ratio_rolling_{w}h`：在时间窗口内情感分数小于 -0.05 的比率。

- `pos_ratio_rolling_{w}h`：在时间窗口内情感分数大于 0.05 的比率。

- `comment_rate_{w}h`：时间窗口内的评论计数（通过对 index 为 DatetimeIndex 的 ones-series 应用 `rolling(window='24H').sum()` 实现）。该计数反映在该时间段内的事件密度，而非行序列位置。

- `engagement_rolling_{w}h`（可选）：若存在 `engagement_score`，计算其在时间窗口内的滚动均值。

- `toxicity_rolling_{w}h`（可选）：若存在 `toxicity_score`，计算其在时间窗口内的滚动均值。

实现备注与建议：

- 当前实现已切换为基于时间偏移的 rolling（如 `rolling('24H')`），因此窗口字符串 `'24H'` 表示向前 24 小时；当你的最小时间粒度为“天”时，`24H` 等价于 `'1D'`（1 天），`48H`≈`2D`，`72H`≈`3D`。

- 如果你想显式以“天”为单位，可以使用 `'1D'/'2D'/'3D'` 这样的窗口字符串；两种写法在大多数情况下是等价的，但 `'1D'` 更直观地表示“日”粒度。

- 由于现在是基于 DatetimeIndex 的时间窗口滚动，请确保输入 DataFrame 的 `review_date` 能被成功解析为 datetime（脚本会尝试 `pd.to_datetime`），并且建议保留原始时间信息（若只有日期部分也可工作，但会默认视为该日的 00:00）。

- 对于非常稀疏或非常密集的时间分布，time-based rolling 的窗口行为与基于样本数的 rolling 会有所不同（time-based 更能反映真实的时间范围内事件密度）。如果你希望先按日/小时汇总再计算滚动，也可以先进行重采样然后调用 `extract()`。


### 3) 差分与加速度特征

- `sentiment_diff_24h`, `sentiment_diff_48h`, `sentiment_diff_72h`：对应窗口滚动均值的差分（.diff()），反映短期变化率（Delta）。

计算： sentiment_diff_24h[t] = sentiment_rolling_mean_24h[t] - sentiment_rolling_mean_24h[t-1]

- `sentiment_acceleration_{w}h`：上述差分的再次差分 (.diff())，代表趋势变化的二阶导数（加速度）。

用途：捕捉情感趋势的拐点与快速变化。


### 4) 评论量变化率

- `comment_rate_change_24h`, `comment_rate_change_48h`, `comment_rate_change_72h`：对应 comment_rate 的 pct_change（百分比变化）。

计算： comment_rate_change_24h = comment_rate_24h.pct_change()

实现注意：若前值为 0 会导致 inf/NaN，最终结果会保持 NaN，需要 downstream 处理（如 fillna(0) 或 clip）。


### 5) 峰值/谷值与趋势方向

- `is_sentiment_local_peak`：布尔值 0/1，若当前点的 `sentiment_rolling_mean_24h` 同时大于前后点则为 1。实现为比较 shift(1) 与 shift(-1)。

- `is_sentiment_local_trough`：相反条件的小于判断。

- `sentiment_trend_direction`：使用 np.sign(sentiment_diff_72h)，得到 -1/0/1，表示近期趋势方向。


### 6) 波动性

- `sentiment_volatility`：定义为 rolling std (72h) 除以 (abs(rolling_mean_72h) + 0.01) 以避免除以零。

用途：衡量情感分布相对于均值的相对波动性。


### 7) 统计异常检测生成的列（`detect_statistical`）

- `is_sentiment_anomaly`：若 |vader_compound - mean| > sigma * std，则标记为 1（sigma 由 sensitivity 设置：low=3, medium=2.5, high=2）。

- `hourly_count`：若不存在，则以 `pd.to_datetime(review_date).dt.floor('H')` 做 groupby 并统计当前小时计数，然后通过 transform('count') 填回每行。

- `is_volume_anomaly`：若 `hourly_count` 超过 mean + sigma*std，则为 1（单侧异常）。

- `is_vote_anomaly`（当存在 `votes_up` 与 `text_length`）：计算 `votes_per_char = votes_up / (text_length + 1)`，若超过 mean + sigma*std，则为 1。

- `is_rare_topic`（当存在 `dominant_topic`）：统计每个主题的频率，若频率 < 0.05 标为稀有主题（1）。

- `is_trend_anomaly`（当存在 `sentiment_diff_24h`）：若 |diff - mean| > sigma*std 标为 1。

- `statistical_anomaly_score`：将所有以 `is_` 开头且包含 `anomaly` 的列相加（整型），表示异常维度数量。

- `is_statistical_anomaly`：当综合分数 >= 2 时标记为 1。



### 8) ML 异常检测（`detect_ml`）

生成或使用如下列：

- `ml_anomaly`：1/0，模型预测为异常时为 1。使用 IsolationForest 或 EllipticEnvelope。
- `ml_anomaly_score`：异常程度分数，脚本使用 -model.score_samples(X)（IsolationForest 返回样本分数，取负值以便大分数表示更异常）。

默认特征集（如果在 DataFrame 中存在）包括：
- `vader_compound`, `sentiment_rolling_std_24h`, `sentiment_diff_24h`, `toxicity_score`, `text_length`, `comment_rate_24h`, `neg_ratio_rolling_24h`, `sentiment_volatility`，以及可选的 `engagement_score`, `votes_up_log`, `comment_count_log`。

实现注意：当前实现使用了更稳健的预处理流水线：`SimpleImputer(strategy='median')` + `StandardScaler()`，并把 imputer 与 scaler 存在 `AnomalyDetector` 实例上以便重用（batch-wise fit+predict 场景）。因此你不再需要在外部单独做 fillna/scale，除非你要求一个预先分离的 fit/predict API（可作为后续改进）。

附加说明（组合检测与分段）：

- `detect_combined(df, ml_weight=0.6, q_medium=0.8, q_high=0.95)`：在运行统计与 ML 检测后，默认会对 `ensemble_anomaly_score` 做简单校准与分段，生成两列：
	- `anomaly_severity_score`：基于 min/max 线性归一的 0-1 分数（用于可视化与阈值比较）。
	- `anomaly_severity`：分段标签，取值为 `'low'`, `'medium'`, `'high'`。默认使用 `q_medium=0.8` (80th 百分位) 与 `q_high=0.95` (95th 百分位) 作为边界。你可以通过参数 `q_medium`/`q_high` 调整这些分位点。

	该 bucketing 是经验性策略，用于无监督场景下把连续异常分数映射为便于告警的离散等级。若有标注数据，建议使用监督校准方法（如 Platt scaling 或 isotonic regression）来得到更可信的概率/等级。



### 9) 协同行为检测（`detect_coordinated`，默认启用）

当 `networkx` 可用时，脚本会：

1. 以指定时间窗口（默认 24 小时）对评论进行分块：`time_window = floor(review_date, '{N}H')`。
2. 在每个窗口内计算 `review_content_clean` 的 TF-IDF（max_features=500），并计算余弦相似度矩阵。
3. 找到相似度高于阈值（默认 0.85）的对，构建带权图并求连通分量。
4. 若连通分量中节点 >= 10 则视为可疑团伙，记录其索引与平均相似度。

输出/标记：
- `is_coordinated`：行级布尔标记（1 表示该评论属于可疑群体）。
- 额外返回 `coordinated_groups`：列表字典，包含 window、indices、size、avg_similarity 等信息，并会被序列化为 `{prefix}_coordinated_groups.json`（默认在与增强 CSV 相同目录保存）。

实现注意：
- 该方法对文本相似度敏感，短文本或清洗不足会导致高误报。TF-IDF 参数（max_features、停用词处理等）可调。
- 若 `networkx` 不存在或者在窗口中文本量不足时（默认窗口内文本数 < 10），函数会跳过并把 `is_coordinated` 设为 0。


## 边界情况与建议

1. 时间窗口的语义：脚本中使用的 rolling window 以样本数为单位（window=w），这要求输入数据在时间上的采样频率稳定。强烈建议在使用前对原始数据进行按小时重采样或以小时为单位的索引来保证 w 对应小时数。

2. 缺失值处理：多数特征在计算前不会显式填充。脚本现在默认在 ML 检测中使用中位数插补（SimpleImputer(strategy='median')）并做 StandardScaler 归一化；但如果在 `process_features_to_full` 中启用了全局统一填补（impute=True），该函数会在合并所有 app 的 trend 特征后执行按列中位数填充（数值）与空字符串填充（非数值）。

3. 数值稳定性：pct_change 会出现 inf/NaN，使用 clip 或 fillna 可避免 downstream 错误。

4. 可解释性：统计异常检测提供了可解释的组合得分（statistical_anomaly_score），适合作为首层过滤；ML 检测提供更灵活但更难解释的结果。


## 推荐的演进/改进方向

- 将 `AnomalyDetector` 的预处理器拆分为可持久化的 `fit` / `transform` API，使预测环境可以复用已训练的 imputer/scaler 与模型。
- 将异常分数进一步分段（low/medium/high）或做校准，以便用于统一告警策略。

---



