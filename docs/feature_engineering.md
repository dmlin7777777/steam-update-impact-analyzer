# feature_engineering.py — 功能与可调参数说明

本文档说明项目中的 `feature_engineering.py` 实现了哪些功能、主要类/方法的作用、可调整的参数（以及默认值）、依赖项、使用示例与注意事项。

## 概览

`feature_engineering.py` 提供了一个以 GPU 优化为导向的特征工程工具类 `GPUOptimizedFeatureEngineer`，用于从游戏评论 / 新闻 / 更新事件数据中抽取丰富的文本和行为特征，内容包括：

- 基本文本统计（长度、词数、句数、标点等）
- 可读性指标（Flesch reading ease, Flesch-Kincaid）
- 情感特征（VADER）及情感类别
- 游戏相关主题词检测（bug 报告、平衡性抱怨、货币化投诉、性能提及）
- 毒性检测（支持 Detoxify 模型的 GPU 加速；否则回退为关键词方法）
- 可选的 Google Perspective API 内容评分（TOXICITY / INSULT / SEVERE_TOXICITY 等），通过 `extract_perspective_features` 提取并可由 `create_comprehensive_features` 的 `use_perspective` 开关启用。
- TF-IDF 特征及稀疏矩阵统计（max/mean/std/nonzero）
- BERT（sentence-transformers）嵌入，可选聚类（MiniBatchKMeans）
- LDA 主题建模（并可做 PCA 降维、dominant topic、topic entropy 等）
 - 基于新闻/更新事件的时间窗口聚合（24/48/72 小时窗口统计）
- 作者/用户行为特征（游戏数、评论数、时长相关比率、活跃标签）
- 句法/语法类特征（spaCy 可选；fallback 为简单启发式）
- I/O 辅助：从目录批量加载 reviews/news、按 genre 批量处理并保存特征

## 主要类与方法（简要）

- `GPUOptimizedFeatureEngineer(...)` — 主类，构造函数参数在下面详细列出。
- `setup_nltk()` — 检查/下载 NLTK 资源（punkt, stopwords, wordnet, vader_lexicon）。
- `setup_models()` — 初始化 TF-IDF、可选的 sentence-transformers、Detoxify、Perspective API 客户端等。
- `_standardize_columns(df)` — 根据内部 `column_aliases` 将 DataFrame 中常见列名映射为规范列名（例如 `review_content_clean`、`review_date`、`appid` 等）。如果缺失会补默认值。
- `extract_text_features(df)` — 提取基本文本统计、标点/大写比例、URL/mention 检测、可读性分数。
- `extract_sentiment_features(df)` — 使用 NLTK VADER 生成 compound / pos / neg / neu 得分与情感类别。
- `extract_game_specific_features(df)` — 基于关键词检测游戏相关话题：bug、balance、monetization、performance。
- `extract_toxicity_features(df, batch_size=32)` — 优先使用 Detoxify（若可用且已加载）；否则使用关键词回退。GPU 时会自动放宽 batch_size。
- `extract_tfidf_features(df, max_features=1000)` — 生成 TF-IDF 矩阵并返回每条样本的聚合统计（max/mean/std/nonzero）。
- `extract_bert_features(df, max_length=256, batch_size=32)` — 用 sentence-transformers 生成 embedding（可在 GPU 上），并返回聚类/距离等统计（若可用）。
- `extract_topic_features(df, n_topics=10, vectorizer='tfidf')` — 使用 LDA 抽取主题分布；支持 `'tfidf'` 或 `'count'` 作为输入向量化器。会生成每个 topic 的概率、dominant topic、topic entropy、topic PCA 等。
- `load_reviews_from_combined_dir(combined_dir, max_rows_per_file=None)` — 批量读取目录下 reviews（csv/xlsx），并尝试从文件名推断 `genre`。
- `load_news_from_dir(news_dir)` — 批量读取 news（csv/xlsx），提取 `appid` 与 `news_date`。
- `align_comments_to_latest_patch(comments_df, news_df, ...)` — 对每条评论找到其之前最近的一次新闻/更新事件并计算小时差（`hours_since_latest_news`）。
- `aggregate_comments_by_news_windows(news_df, comments_df, windows=[24,48,72])` — 为每条 news 统计事件窗口内的评论数/平均情感/负/正比率。
- `aggregate_time_window_features(comments_df, windows=[24,48,72], ...)` — 基于 `hours_since_latest_news` 为每条评论生成窗口级统计并按分组键（如 `genre`/`appid`）合并。
- `extract_author_features(df)` — 生成作者相关特征（游戏/评论比、小时/游戏比、活跃/硬核/休闲标记、author_reputation_score 等）。
- `extract_sentiment_interaction_features(df)` — 计算 per-app/genre 的情感均值/std 并生成交互特征（情感×作者声望、情感×bug）。
- `extract_syntax_features(df)` — 如安装 spaCy 则用其 POS 输出比率、否定词/强化词统计；没安装则用简单启发式 fallback。
- `extract_perspective_features(df, batch_size=16)` — 使用 Google Perspective API 为每条评论返回属性评分（TOXICITY/INSULT/SEVERE_TOXICITY/THREAT/IDENTITY_ATTACK），仅当 `perspective_client` 已初始化时才会调用；若不可用则会跳过并返回原始 DataFrame。
- `create_comprehensive_features(df, news_df=None, use_news_events=False, ...)` — 主流程：按顺序运行上述步骤并返回包含 `dataframe`, `tfidf_matrix`, `tfidf_feature_names`, `bert_embeddings`, `feature_summary` 的字典。注意：时间特征已改为“事件中心化”，必须传入 `news_df` 并把 `use_news_events=True`。
    - 新增参数 `use_perspective`（默认 False）：若设为 True 且构造器传入了有效的 `perspective_api_key` 且 `googleapiclient` 可用，则会在语法特征之后调用 `extract_perspective_features`，并把 Perspective 返回的分数列加入最终的 DataFrame。
- `save_features(results, output_dir, file_prefix='features')` — 将生成的 DataFrame/TF-IDF/BERT embedding/summary 等保存到磁盘（CSV/NPZ/NPY/PKL/文本等）。

## 构造器与可调整参数（归纳）

构造器：

`GPUOptimizedFeatureEngineer(perspective_api_key: Optional[str] = None, force_cpu: bool = False, load_heavy_models: bool = True, lda_vectorizer: str = 'tfidf', serialized_models_dir: Optional[str] = None, serialized_models_prefix: Optional[str] = None)`

- `perspective_api_key`：可选，若使用 Google Perspective API 做内容评分，传入 key。
- `force_cpu`：强制使用 CPU（即便能检测到 GPU）。默认 False。
- `load_heavy_models`：是否在初始化时尝试加载大型模型（sentence-transformers、Detoxify、Perspective 客户端等）。默认 True。
- `lda_vectorizer`：LDA 输入的向量化方法（'tfidf' 或 'count'），默认 `'tfidf'`。
- `serialized_models_dir` / `serialized_models_prefix`：若提供，会尝试自动加载序列化的 PCA/KMeans 模型（以便复用已有聚类/PCA）。

类内部常见可调整参数（在对应方法中传入）：

- `extract_toxicity_features(batch_size=32)`：调整批量大小，GPU 情况下默认会放宽至 64。
- `extract_tfidf_features(max_features=1000)`：TF-IDF 特征的最大数量。
- `extract_bert_features(max_length=256, batch_size=32)`：BERT 截断长度与批量大小（GPU 下允许更大 batch）。
- `extract_topic_features(n_topics=10, vectorizer='tfidf')`：LDA 的主题数与向量器类型（'tfidf' 或 'count'）。
 - `create_comprehensive_features(..., use_news_events: bool = False, event_windows: List[int] = [24,48,72], include_bert: bool = True, include_topics: bool = True, n_topics: int = 8, tfidf_features: int = 500, max_bert_length: int = 256, bert_batch_size: int = 32, toxicity_batch_size: int = 32, include_flagged: bool = False)`：主流程中大量可配置项，推荐在调用时显式设置以控制内存/性能。说明：`include_flagged` 控制是否在特征提取阶段包含 `is_removed_cleaning==1` 的行。
- `use_perspective: bool = False`：当为 True 时并且已提供 `perspective_api_key`，会启用对每条评论的 Perspective API 分数提取（注意配额与速率限制）。

另外还有 I/O 与保存相关的可调项：

- `load_reviews_from_combined_dir(combined_dir, max_rows_per_file=None)`：可以限制每个文件读取的行数以降低内存压力。
- `process_and_save_news_level(reviews_dir, news_dir, output_path, max_rows_per_file=None, windows=[24,48,72])`：执行 news-level 聚合并保存（Parquet 或回退到 CSV）。

## 输入数据要求与列名兼容性

该工具对列名具有耐受性：

- 内部维护 `column_aliases` 字典，会将常见别名映射为 canonical 列名（如 `review_content_clean`, `review_date`, `appid`, `genre` 等）。
- 如果某些数值列（如 `num_games_owned`, `num_reviews`, `played_hours`）缺失，会被填充为 0；其他文本列可能被填为 NaN。

推荐至少确保每条评论包含：文本（任一 content 列能被映射到 `review_content_clean` 或 `review_content_processed`）和日期（可映射到 `review_date`）。如果需要时间窗口/事件特征，必须提供 `news_df`（包含 `appid` 与 `news_date`）。

## 输出（create_comprehensive_features 返回值）

返回一个字典：

- `dataframe`：含所有派生特征的 Pandas DataFrame。
- `tfidf_matrix`：scipy.sparse 的 TF-IDF 矩阵（或 None）。
- `tfidf_feature_names`：TF-IDF 特征名数组。
- `bert_embeddings`：numpy.ndarray 的 embedding（或 None，如果 BERT 未加载）。
- `feature_summary`：由 `_create_feature_summary` 生成的类别化特征清单与统计信息。

`save_features` 会将上述 artefacts 持久化到指定目录，文件名包含 `file_prefix`。

## 依赖项（最小建议）

- Python 基础： `pandas`, `numpy`, `glob`, `pickle`, `logging`, `re` 等
- NLP / ML： `nltk`, `scikit-learn`, `textstat`, `sentence-transformers`, `torch`（若使用 GPU/BERT）
- 可选但推荐： `detoxify`（毒性检测）、`googleapiclient`（Perspective API）、`spacy` + `en_core_web_sm`（句法特征）、`scipy`（保存稀疏矩阵）、`pyarrow`/`fastparquet`（parquet 写入）
    - 如果要使用 `use_perspective`，请安装 `google-api-python-client`（`pip install google-api-python-client`）并确保在实例化 `GPUOptimizedFeatureEngineer` 时传入有效的 `perspective_api_key`（建议通过环境变量传入）。

示例安装（PowerShell）：

```powershell
python -m pip install -U pip
python -m pip install pandas numpy nltk scikit-learn textstat sentence-transformers torch scipy
# 可选组件
python -m pip install detoxify google-api-python-client spacy pyarrow openpyxl
python -m spacy download en_core_web_sm
```

注意：`sentence-transformers` 会自动安装较多依赖并可能需要 CUDA 对应版本的 `torch` 才能在 GPU 上工作。

## 使用示例

作为脚本运行（仓库提供的 `main()` 在 `scripts/feature/feature_engineering.py`，按文件/genre 批量处理并把特征保存到 `features/` 的对应子目录）：

```powershell
# 在项目根目录运行
python .\scripts\feature\feature_engineering.py
```

作为模块调用（示例）：

```python
from feature_engineering import GPUOptimizedFeatureEngineer
import pandas as pd

fe = GPUOptimizedFeatureEngineer(force_cpu=False, load_heavy_models=True)
reviews_df = pd.read_csv('path/to/reviews.csv')
news_df = fe.load_news_from_dir('path/to/news_dir')
results = fe.create_comprehensive_features(reviews_df, news_df=news_df, use_news_events=True,
                                          include_bert=True, include_topics=True,
                                          n_topics=10, tfidf_features=1500, use_perspective=False)
df_features = results['dataframe']
fe.save_features(results, output_dir='output/features', file_prefix='example')
```

## 常见可调策略与性能建议

- 若内存受限或没有 GPU：设置 `force_cpu=True`、`load_heavy_models=False`，并在 `create_comprehensive_features` 中把 `include_bert=False`、`include_topics=False` 或减小 `tfidf_features`。
- BERT/Detoxify 在 GPU 上能显著加速：确保 `torch` 与 CUDA 版本匹配，并通过 `force_cpu=False` 让类自动使用 GPU。
- 对非常大的数据集，使用 `load_reviews_from_combined_dir(..., max_rows_per_file=...)` 逐文件采样或在 `main()` 中按文件采样子集。

## 已知行为与注意点

- 时间特征已“事件中心化”：`create_comprehensive_features` 要求提供 `news_df` 并设置 `use_news_events=True`，否则会抛出 ValueError。
- 如果 Detoxify / sentence-transformers 等无法加载，代码会退回到更简单的方法（关键词检测、跳过 BERT）。
- `extract_topic_features` 的 `vectorizer` 参数会决定 LDA 使用 TF-IDF 还是 CountVectorizer；LDA 更常与 CountVectorizer 配合使用以符合概率模型假设。

## 建议的改进（可选）

- 将 I/O 与 feature pipeline 拆成更小的可插拔组件（便于单元测试与流水线复用）。
- 将 model-loading 的细节（model name、device、batching 策略）暴露为更细粒度的构造器参数或配置文件。
- 添加单元测试覆盖关键方法（尤其是时间对齐、topic 生成与保存/加载序列化模型）。

---

如果你希望我把该文档再扩展为 README 的一部分、把依赖列成 `requirements.txt`，或把示例代码写成小脚本/测试用例，我可以继续帮你实现这些改动。

## Runner CLI 文档（`scripts/run_fe_trend_anomaly.py`）

该脚本用于按 appid 分组执行：Feature Engineering → Trend Extraction → Anomaly Detection，并把最终增强特征保存为 CSV。行为重要更新：脚本/工具会递归地在 `features/` 下发现由 FE 生成的 per-genre `*_dataframe.csv` 文件并分别处理，输出的增强特征和 FE artifacts 会保存在与输入 FE 文件相同的文件夹中（便于按 genre 管理）。

- `--include-flagged`：默认不启用。若指定，则在 FE 阶段包含被标记为 `is_removed_cleaning==1` 的行（这些通常被认为是垃圾或机器人评论）。
- `--save-features`：默认不启用。若指定，脚本将在 FE 输入文件所在的同一目录保存 FE 的 artifacts（DataFrame CSV、TF-IDF 矩阵 `.npz`、TF-IDF 特征名 `.txt`、BERT embeddings `.npy`、summary `.txt` 等），并在文件名中加入 `_inclflagged` / `_exclflagged` 表示是否包含被标记的行。

常用示例（PowerShell）：

1) 默认运行（使用仓库默认 reviews 路径，排除被标记行）：

```powershell
python .\scripts\run_fe_trend_anomaly.py --out "features\combined\enhanced_exclflagged.csv"
```

2) 使用无 label reviews 并保存 FE artifacts（排除被标记行）：

```powershell
python .\scripts\run_fe_trend_anomaly.py --reviews "c:\Users\12932\Desktop\nus\BAP\data_nolabel\combined" --news "c:\Users\12932\Desktop\nus\BAP\data_nolabel\cleaned\news" --out "features\combined\enhanced_exclflagged.csv" --save-features
```

3) 包含被标记行并保存 FE artifacts（用于审计与对比）：

```powershell
python .\scripts\run_fe_trend_anomaly.py --reviews "c:\Users\12932\Desktop\nus\BAP\data_label\combined" --news "c:\Users\12932\Desktop\nus\BAP\data_nolabel\cleaned\news" --out "features\combined\enhanced_inclflagged.csv" --include-flagged --save-features
```

注意：脚本现在按发现到的 FE 输入文件在其所在目录保存增强特征；如果使用 `--save-features`，FE artifacts 会写入同一目录并带上 `_inclflagged` / `_exclflagged` 后缀以便区分。若你想仅处理某些 genre，请把相应 FE 文件移动或指定文件夹。
