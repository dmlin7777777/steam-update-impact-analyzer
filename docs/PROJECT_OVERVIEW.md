 # 项目概述 — 特征工程、趋势与异常检测

 本文档为仓库中与特征工程、趋势/异常检测及汇总分析相关的代码提供中文说明，包含模块职责、默认路径与文件命名约定、运行与扩展建议，以及常见问题排查要点。

 ## 高层流程

 1. 数据清洗（独立脚本）：清理评论/新闻数据并输出清洗后的文件。
 2. 特征工程（`scripts/feature/feature_engineering.py`）：从文本与元数据中提取情感、主题、参与度、作者信息等特征，并将每个类别（genre）的特征文件保存到 `features/` 目录下。
 3. 趋势与异常处理（`scripts/feature/trend_anomaly_analysis.py`）：读取特征文件，计算基于时间窗口的趋势特征（24/48/72 小时窗口），执行统计/机器学习异常检测与协调行为检测（coordinated detection），并输出“增强型”CSV 与可选的协调组 JSON 报告。
 4. 汇总/分析（`scripts/summary/feature_analyzer.py`）：加载特征 CSV，进行探索性分析并提供简单的模型训练工具与报告导出。

 ## 主要文件与职责

 - `scripts/feature/feature_engineering.py`
   - 类：`GPUOptimizedFeatureEngineer`
   - 职责：
     - 规范输入列并提取丰富特征：文本统计、TF-IDF、情感（VADER）、可选的 BERT 嵌入、LDA 主题与 topic-PCA、toxicity（Detoxify 或关键字回退）、作者与参与度特征等。
     - 主要 API：`create_comprehensive_features(...)`，用于从 DataFrame 生成特征（接受 `include_flagged` 参数以控制是否包含清洗时标记为已移除的行）。
     - 按类别保存特征文件，文件名后缀含义为是否包含被清洗标记（`_inclflagged` / `_exclflagged`）。
     - 自动加载序列化的模型（topic PCA / embedding KMeans）：支持通过 `preloaded_topic_pca_path` / `preloaded_embedding_kmeans_path` 指定精确路径，若未指定则按目录 + 前缀模式查找；支持 `pickle` 与 `joblib` 两种序列化格式。
     - 默认时间窗口：`[24, 48, 72]` 小时。

 - `scripts/feature/trend_anomaly_analysis.py`
   - 关键类：`TrendFeatureExtractor`、`AnomalyDetector`、`ComprehensiveAnalysisSystem`
   - 职责：
     - `TrendFeatureExtractor.extract(df)`：基于小时级时间窗口（24/48/72h）计算滚动均值、差分、加速度、波动性等趋势特征，用于后续分析或外部建模。
     - 注意：仓库中原先用于训练与预测的 `TrendForecaster`（LightGBM 相关逻辑）已从本模块中移除，当前模块不再负责模型训练/预测。
     - `AnomalyDetector`：实现统计学检测、基于模型的检测（IsolationForest、EllipticEnvelope 等）以及基于图的协调行为检测（使用 NetworkX + TF-IDF + 余弦相似度）。
     - `process_features_to_full(output_dir, file_prefix, include_flagged=None, ...)`：加载特征工件，按 `appid` 聚合并计算趋势特征，执行异常检测，输出增强型 CSV（格式 `{file_prefix}{_inclflagged/_exclflagged}_enhanced_features.csv`），并默认执行协调检测（如发现协调组则写出 `{prefix}_coordinated_groups.json`）。
     - `main()`：递归查找默认 `features` 根目录（默认：`C:\Users\12932\Desktop\nus\BAP\features`）下的 `*_dataframe.csv` 文件并逐个处理。

 - `scripts/summary/feature_analyzer.py`
   - 类：`FeatureAnalyzer`
   - 职责：
     - 加载特征 CSV，执行探索性分析并提供简单的模型训练辅助（如评论质量预测、toxicity 检测），并可导出洞察报告。
     - 示例 `main()` 会在 `C:\Users\12932\Desktop\nus\BAP\features` 下递归查找 `*_dataframe.csv` 并运行分析（若找到）。

 ## 重要默认值与文件位置

 - 特征根目录（默认）：`C:\Users\12932\Desktop\nus\BAP\features`
   - 仓库期望把每个类别的特征文件存放在该目录的子目录中（按 genre 分类），代码会递归发现这些文件。
 - 特征工件命名约定（由 `save_features` 生成）：
   - `{file_prefix}_dataframe.csv`（通用）
   - `{file_prefix}_inclflagged_dataframe.csv`（包含被清洗标记的行）
   - `{file_prefix}_exclflagged_dataframe.csv`（排除被清洗标记的行）
 - 增强输出（趋势/异常）：
   - `{file_prefix}{suffix}_enhanced_features.csv` — 保存在特征文件相同目录下，`suffix` 为 `_inclflagged`、`_exclflagged` 或空（未知情况）。
   - `{file_prefix}_coordinated_groups.json` — 若检测到协调群体则输出的 JSON 报告。
 - 模型持久化：
   - 说明：本模块已移除内置的模型训练与持久化功能（原 `TrendForecaster` 已删除）。若需要进行趋势预测训练或持久化模型，请在外部脚本中使用 `TrendFeatureExtractor` 生成的特征数据，然后独立训练与保存模型（例如使用 `joblib` 或其他序列化方式）。
   - `feature_engineering` 的自动加载逻辑仍会先尝试 `pickle`，若失败则回退到 `joblib`，并在尝试显式 `preloaded_*` 路径时优先加载这些文件。

 ## 重要运行参数（API）

 - `GPUOptimizedFeatureEngineer(..., serialized_models_dir=None, serialized_models_prefix=None, preloaded_topic_pca_path=None, preloaded_embedding_kmeans_path=None)`
   - 当你有确切的模型路径时使用 `preloaded_*` 参数；否则可提供 `serialized_models_dir` 与（可选）`serialized_models_prefix` 让代码自动搜索。
 - `create_comprehensive_features(df, include_flagged: bool = False, ...)`
   - 控制是否在特征生成阶段包含被清洗标记为移除（flagged）的行。
 - `process_features_to_full(output_dir, file_prefix, include_flagged=None, windows=None, save_path=None, run_coordinated: bool = True)`
   - `include_flagged` 可设为 True / False / None（自动判断）；`windows` 默认为 `[24,48,72]`；`run_coordinated` 默认为 True（除非显式设为 False，否则会运行协调检测）。

 ## 依赖项

 - 建议安装（必需/推荐）：
   - pandas、numpy
   - scikit-learn
   - lightgbm
   - sentence-transformers（可选，当 `load_heavy_models=True` 且启用 BERT 特征时使用）
   - joblib
   - textstat、nltk
   - matplotlib、seaborn（可视化）
 - 可选（增强功能）：
   - detoxify（toxicity 模型回退）
   - google-api-python-client（Perspective API）
   - networkx（用于协调行为检测）

 注意：如果未安装 `networkx`，协调检测会记录警告，且 `is_coordinated` 列的值将全部为 0（无群体检测）。

 ## 性能与扩展注意事项

 - BERT 嵌入与 Detoxify 模型比较耗 GPU/内存；`GPUOptimizedFeatureEngineer` 会检测 CUDA 并在可用时使用 GPU。
 - 协调检测（TF-IDF + 余弦相似度 + 图的连通分量）在窗口内的文本比较为 O(n^2)，当候选数量大时计算代价高。可采用抽样、子窗口化或只对候选时段运行协调检测以降低成本。
 - 建议按类别保存特征，`trend_anomaly_analysis.main()` 会对每个类别目录分别处理，从而限制单次作业的数据量。

 ## 常见问题与排查

 - “我的 PCA/KMeans 无法自动加载”：
   - 可能是使用了与期望不同的序列化格式（joblib vs pickle），或文件名与前缀不匹配。可通过 `preloaded_topic_pca_path` 与 `preloaded_embedding_kmeans_path` 提供精确路径，或将序列化文件放入 `serialized_models_dir` 并使用可识别的前缀。
 - “协调检测没有结果”：
   - 请确认已安装 `networkx`。若已安装，请检查时间窗口参数与相似度阈值是否适合你的数据。
 - “内存不足 / 运行慢”：
   - 禁用 BERT 特征或降低 TF-IDF 的 `max_features`，或对数据做采样。对于协调检测，可在调用 `process_features_to_full` 时将 `run_coordinated=False`。

 ## 常见运行示例

 - 按类别生成特征（示例来自 `feature_engineering.main()`）：
   - 输入：`data_label/combined/*`（按类别的合并评论文件）与 `data_nolabel/cleaned/news`（新闻事件）
   - 输出：`features/<genre>/gpu_optimized_features_<genre>_exclflagged_dataframe.csv`（默认排除清洗标记的行）

 - 对单个特征工件运行趋势/异常增强处理：
   ```python
   from scripts.feature.trend_anomaly_analysis import process_features_to_full
   df_out, used_flagged = process_features_to_full(r"C:\...\features\fps", "gpu_optimized_features_fps", include_flagged=False)
   ```

 - 对所有特征进行批量处理（`trend_anomaly_analysis.py` 的 `main` 会在默认 features 根目录下递归发现并处理所有 FE 文件）：
   - 可直接以脚本方式运行，也可以通过 import 调用 `main()`。

 ## 推荐的后续改进

 - 为 `trend_anomaly_analysis.main()` 添加更完善的 CLI 参数（例如 `--features-root`、`--no-coordinated`、指定要处理的子目录、并行度/worker 数量等）。
 - 若 FE 工件包含稳定的 `review_id` 或 `comment_id`，可将 `coordinated_groups` 保存为 CSV 并映射回评论 ID，便于人工验证。
 - 为 `process_features_to_full` 与 `detect_coordinated` 添加单元测试（happy path + 小型合成样本）。
 - 添加 `requirements.txt`，将关键库版本固定（例如 `lightgbm`、`sentence-transformers`、`joblib`、`networkx`）。

 ---

 文件已更新：`docs/PROJECT_OVERVIEW.md`（已替换为中文）

 如果需要，我可以：
 - 打开并在编辑器中显示该文件并根据你的偏好作词句调整；或
 - 生成 `requirements.txt`（最小依赖/推荐依赖）；或
 - 为 `trend_anomaly_analysis.py` 添加 CLI wrapper（包括 `--no-coordinated` 和 `--features-root`）。告诉我你想要哪个下一步。