# trend_anomaly_analysis.py — 功能说明与实现清单

本文档概述 `scripts/feature/trend_anomaly_analysis.py` 当前实现的功能、所使用的内部/外部模块、输入输出约定、如何调用（API 与脚本）、已知限制与改进建议。该文档基于当前仓库版本的源码自动整理。

## 主要功能（概览）

1. 时序趋势特征提取（TrendFeatureExtractor）
   - 按评论时间序列计算多尺度滚动窗口统计（默认窗口为 24、48、72 小时）：情感平均/标准差/最小/最大、正/负比率、评论量、互动与毒性滚动均值等。
   - 生成差分与二阶差分特征（sentiment_diff_* 与 sentiment_acceleration_*），以及评论量变化率（pct change）。
   - 生成本地峰值/谷值、趋势方向（sign）与波动率等信号。

2. 趋势预测（已移除）
   - 说明：仓库中原先包含的 `TrendForecaster`（用于基于滚动窗口特征训练 LightGBM 模型以预测未来情感/评论率）的训练与预测相关代码已被移除。
   - 目前 `trend_anomaly_analysis.py` 的职责已简化为：
     - 计算与导出基于时间窗口的趋势特征（见第 1 项），用于后续分析或外部建模。
     - 执行异常检测（统计与 ML）与协调行为检测（coordinated detection）。
   - 如果需要再次进行趋势预测训练/持久化，请使用外部脚本或调用其他模块来加载由 `TrendFeatureExtractor` 生成的特征并训练模型。

3. 异常检测（AnomalyDetector）
   - 统计规则检测（基于 sigma 阈值的情感/流量/互动/主题异常判断）。
   - ML 检测：IsolationForest（或 EllipticEnvelope）用于无监督异常打分与检测。
   - 协同刷评检测：基于短时间窗口内文本相似度（TF-IDF + cosine similarity）构建图并发现高度相似的连通分量（需 `networkx` 可用）。
   - 输出多种异常标记列（`is_statistical_anomaly`, `ml_anomaly`, `is_coordinated` 等）与异常分数。

4. 综合分析系统（ComprehensiveAnalysisSystem）
   - 封装趋势提取、目标创建、趋势预测训练与异常检测，提供 `analyze()` 返回统一的字典结果（trend alerts、anomaly、coordinated 等）与聚合报告生成函数。

5. 批量处理与增强特征保存
   - 提供 `load_feature_dataframe_from_dir`、`process_features_to_full` 等帮助函数：可从 feature-engineering 输出目录加载 `{prefix}_dataframe.csv`（支持 `_inclflagged` / `_exclflagged` 变体优先级），对每个 `appid` 分组进行趋势提取与异常检测，保存增强后的 CSV（按 `prefix` 与变体生成 `{prefix}{_inclflagged/_exclflagged}_enhanced_features.csv`）。
   - `main()` 会递归查找 `features/**/*_dataframe.csv`（per-genre），并在每个文件所在目录分别处理可用的变体（excl/incl/generic）。

## 使用到的库与依赖

- 必需（代码中直接 import）：
  - pandas, numpy, logging, glob, os, datetime
  - scikit-learn（IsolationForest, EllipticEnvelope, TfidfVectorizer, cosine_similarity, StandardScaler, TimeSeriesSplit）
  - lightgbm
  - matplotlib, seaborn

- 可选（存在 try/except）：
  - networkx（用于协同刷评检测；若不可用相应检测将被跳过）

## 输入与输出约定

- 输入 DataFrame（用于趋势/异常分析）至少应包含列：
  - `review_date`（可解析为 pandas datetime）
  - `vader_compound`（情感分数）
  - `appid`（分组键，用于按游戏分组）
  - 可选列：`engagement_score`, `toxicity_score`, `review_content_clean`, `votes_up`, `text_length` 等会启用额外检测/特征。

- FE artifact 文件格式（由 `feature_engineering.save_features` 产生并由本模块读取）：
  - `{prefix}_dataframe.csv` 或 `{prefix}_inclflagged_dataframe.csv` / `{prefix}_exclflagged_dataframe.csv`
  - 本模块 `process_features_to_full` 会输出 `{prefix}{suffix}_enhanced_features.csv` 到相同目录下，suffix 为 `_inclflagged` / `_exclflagged` 或空串（若无法确定）。

## 主要函数 / API（快速参考）

- TrendFeatureExtractor(windows=[24,48,72]).extract(df) -> df_with_trend
- AnomalyDetector(method='isolation_forest').detect_ml(df) / detect_statistical(df) / detect_coordinated(df)
- ComprehensiveAnalysisSystem.analyze(df) -> results_dict  (注意：本系统已去除自动训练/预测功能，返回以趋势特征与异常结果为主的分析字典)
- load_feature_dataframe_from_dir(output_dir, file_prefix, include_flagged=None) -> (df, used_flagged)
- process_features_to_full(output_dir, file_prefix, include_flagged=None, windows=None) -> (df_out, used_flagged)
- main() — 脚本入口，递归处理 `features/**` 下的 FE artifact 文件并保存增强特征到同一目录

## 限制与已知问题

- 依赖：LightGBM 与 scikit-learn 要与本地环境兼容，部分函数在 Windows + multiprocessing/threads 下可能受限（代码已在 LDA/parallel 处使用 n_jobs=1 写法以避免序列化问题）。
- 数据完整性：`vader_compound` 与 `review_date` 为核心列，缺失会导致异常或抛错。
- 协同检测依赖 `networkx`，若未安装则检测被跳过并会产生命名为 `is_coordinated = 0` 的列。
- `main()` 会递归查找 `features/**`，若项目中存在大量无关 CSV 匹配模式会增加无效扫描——可通过将 FE artifacts 放在专门子目录或在 `main()` 调用中传入特定目录来限制。

## 建议改进

- 增加 `process_features_to_full(..., save=False)` 以便在内存中运行而不写文件（用于单元测试或交互式分析）。
- 将 `main()` 的默认路径与扫描深度暴露为 CLI 参数以便更精确控制处理范围。
- 为 `load_feature_dataframe_from_dir` 添加更多容错：检查 header/列名不一致并尝试自动映射。
- 为关键流程添加单元测试（TrendFeatureExtractor.extract、process_features_to_full 的 happy path 与 boundary cases）。

---

如果你要我把这个文档保存到 repo 的 README 或把它扩展为更详细的开发者指南（含运行命令、示例输入/输出片段与常见故障排查），我可以继续完成这些工作。