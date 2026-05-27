# 特征说明（中文）

本文档总结并说明了 `feature_engineering.py`（GPUOptimizedFeatureEngineer）生成的所有特征、计算方法、输入要求、缺失值处理和注意事项。

目录
- 概述
- 通用前提
- 文本特征
- 情感特征
- 游戏特定特征
- 毒性与不当言论特征
- Perspective API 特征（可选）
- TF-IDF 特征
- BERT 嵌入相关特征
- 主题（LDA）特征
- 时间/事件窗口特征
- 作者/用户特征
- 评论互动与影响力特征
- 语法/句法特征
- 交互特征
- 保存与输出
- 建议的改进

---

## 概述

`GPUOptimizedFeatureEngineer` 提供了一整套用于游戏评论数据的特征提取函数。主要入口为 `create_comprehensive_features`，它按以下顺序调用各类提取函数并返回一个包含加工后 DataFrame 与中间矩阵（如 TF-IDF、BERT embeddings）等的结果字典。

主要依赖项（部分为可选）:
- pandas, numpy, nltk, textstat, sklearn, sentence-transformers, torch
- 可选: Detoxify（毒性检测），Perspective API 客户端

通用列名（脚本会尝试标准化多种别名）:
- review_content_clean / review_content_processed: 评论文本（清洗或预处理后）
- review_date: 评论时间（将被转换为 pandas datetime）
- appid: 游戏 id
- votes_up, votes_funny, comment_count, weighted_vote_score, voted_up: 互动相关列

## 文本特征（extract_text_features）

这些特征基于 `review_content_clean`。如果缺失，许多函数会将其视为空字符串或 NaN。

- text_length: 文本字符长度（len）
- word_count: 以空白拆分的词数
- sentence_count: 使用 NLTK 的 sent_tokenize 计算句子数
- avg_word_length: 平均词长 = 所有词长度的均值；若无词则为 0
- exclamation_count: 文中 '!' 的数量
- question_count: 文中 '?' 的数量
- uppercase_ratio: 文中大写字符数 / 总字符数，若文本为空则为 0
- punctuation_ratio: 文中常见标点(.,!?;:)的比例
- readability_score: 使用 textstat.flesch_reading_ease 的可读性得分；空字符串返回 0
- flesch_kincaid_grade: 使用 textstat.flesch_kincaid_grade
- has_question: question_count > 0 的二值指示器（0/1）
- has_url: 是否包含 URL（正则检测）
- has_mention: 是否包含 @mention（正则检测）
- has_external_reference: has_url 或 has_mention

注意事项：文本必须为字符串；函数对空值使用填充/默认值以避免异常。

## 情感特征（extract_sentiment_features）

基于 NLTK VADER 情感分析（`review_content_clean`），对于每条评论计算：

- vader_compound: VADER 的 compound 分数（范围 -1 到 1）
- vader_positive: positive 成分分数
- vader_negative: negative 成分分数
- vader_neutral: neutral 成分分数
- sentiment_category: 基于 compound 的分类：>=0.05 -> 'positive'；<=-0.05 -> 'negative'；否则 'neutral'
- sentiment_label: 数值化标签（negative=0, neutral=1, positive=2）

注意：VADER 对英文短文本效果好；对其他语言或特殊用语需谨慎。

## 游戏特定特征（extract_game_specific_features）

使用关键字匹配检测评论是否包含以下类别信息（均为 0/1）：

- contains_bug_report: 包含 crash, bug, freeze, lag, fix, patch 等词
- contains_balance_complaint: 包含 op, overpowered, nerf, buff, meta, cheater, toxic 等词
- contains_monetization_complaint: 包含 pay to win, p2w, paywall, microtransaction, dlc, money grab 等词
- mentions_performance: 包含 fps, framerate, performance, optimization, graphics 等词

实现细节：使用 \b 单词边界的正则，忽略大小写，缺失文本处理为 False。

## 毒性与不当言论特征（extract_toxicity_features / _extract_toxicity_keywords）

该模块首先尝试使用 Detoxify（若可用且成功加载）以 batch 方式预测毒性分数（模型输出的 'toxicity'），否则回退到基于关键字的方法：

基于 Detoxify 时：
- toxicity_score: Detoxify 返回的毒性分数（原始概率/置信度）
- is_toxic: toxicity_score > 0.5 时为 1，否则 0

基于关键字时（回退）：未使用
- contains_toxic_keywords: 是否匹配到 hate, stupid, idiot, trash, suck 等词（0/1）
- toxicity_score: contains_toxic_keywords * 0.7（简化评分）
- is_toxic: same as contains_toxic_keywords

批处理与 GPU：如果 GPU 可用，会尝试在更大 batch_size 下运行并记得清理 GPU 缓存。

## Perspective API 特征（extract_perspective_features，可选）未使用

当并且仅当提供 `perspective_api_key` 且安装了 googleapiclient 时，函数会对每条评论调用 Google Perspective API，提取下列属性（列名以 perspective_ 前缀）:
- perspective_toxicity
- perspective_severe_toxicity
- perspective_insult
- perspective_threat
- perspective_identity_attack

实现细节：逐条调用并短暂 sleep（默认 0.1s）以控制速率，失败时保留 NaN。

## TF-IDF 特征（extract_tfidf_features）

基于 `review_content_processed`（预处理文本）构建 TF-IDF 矩阵并返回若干统计特征：

- tfidf_max: 文档中最大的 TF-IDF 值（稀疏矩阵安全计算）
- tfidf_mean: 行均值（sum / n_features）（包含对零的平均）
- tfidf_std: 对行的标准差（基于 mean of squares - square of mean）
- tfidf_nonzero_count: 文档中非零 TF-IDF 特征数量（getnnz）

返回值：函数返回 (df_with_features, tfidf_matrix)。

实现要点：会根据样本量动态设置 min_df / max_df，并保证数值稳定。

## BERT 嵌入相关特征（extract_bert_features）

使用 sentence-transformers（默认 'all-MiniLM-L6-v2'）计算嵌入。函数会：

- 对文本做截断（max_length，默认 256 字符）以节省 GPU 内存
- 批量编码并返回 embeddings 矩阵（N x D，D ≈ 384）

并派生若干统计特征加入 DataFrame：
- bert_embedding_norm: 向量范数（L2）
- bert_embedding_mean: 向量均值（各维度平均）
- bert_embedding_std: 向量标准差

还会尝试生成 embedding 聚类特征（通过 _generate_embedding_clusters）:
- embed_cluster_id: 聚类标签（整数）
- embed_cluster_dist: 到簇中心的 L2 距离

如果 sentence-transformers 无法加载，会跳过并返回 None。

注意：函数以 normalize_embeddings=True 生成单位向量，因而 norm 通常接近 1（除非 fallback 用 0 向量）。

## 主题特征（LDA, extract_topic_features）

使用 LDA 提取主题分布（可选基于 TF-IDF 或 CountVectorizer）：使用CountVectorizer

生成的特征包括：
- topic_{i}_prob: 每个主题 i 的概率（i=0..n_topics-1）
- dominant_topic: 具有最大概率的主题索引
- dominant_topic_prob: 最大主题概率
- topic_entropy: 主题分布的熵（-sum p log p）
- topic_pca_{j}: 将主题分布通过 PCA 降维后的连续特征（默认 k=min(5,n_topics)）
- topic_top2_pair: top2 主题的组合字符串（例如 t1_t3）
- topic_top2_prob_product: top1_prob * top2_prob
- topic_category: 自动映射的主题标签（尝试与预定义标签相似度匹配）

实现细节：
- LDA 使用 sklearn.decomposition.LatentDirichletAllocation（n_jobs=1 on Windows）
- 若 self.topic_pca 已存在则 reuse，否则 fit 并保存
- 从 LDA components 中抽取 top N 关键词给主题命名（使用当前 tfidf_vectorizer 的词表）

注意：LDA 在小数据集上可能不稳定，建议在训练集上预训练 LDA 与 PCA 并保存以供推理时加载。

## 时间/事件窗口特征与对齐（align_comments_to_latest_patch, aggregate_time_window_features 等）

核心思想：将每条评论对齐到同一 appid 下最新（且早于评论时间的）news/update 时间，计算评论相对于该更新发生的小时数（hours_since_latest_news），并基于该小时数在多个窗口（例如 24/48/72 小时）内聚合评论统计。

生成列：
- latest_news_date: 评论之前最近的 news_date（或 NaT）
- hours_since_latest_news: 评论时间 - latest_news_date（小时，>=0；找不到 prior update 则为 NaN）

基于窗口的聚合（aggregate_comments_by_news_windows / aggregate_time_window_features）：每个窗口 w（小时）会生成：
- count_in_{w}h: 窗口内评论数（或分组下的计数）
- avg_sent_in_{w}h 或 avg_sentiment_in_{w}h: 窗口内平均情感（vader_compound）
- neg_ratio_in_{w}h: 窗口内负面评论比例（compound <= -0.05）
- pos_ratio_in_{w}h: 窗口内正面评论比例（compound >= 0.05）

实现要点：
- align_comments_to_latest_patch 首选使用 pandas.merge_asof（更快），若失败回退到逐行查找
- aggregate_time_window_features 支持按 'genre' 或 'appid' 分组，如找到会做 groupby 聚合，否则做全局统计

## 作者/用户特征（extract_author_features）

基于作者的历史元数据（如 num_games_owned, num_reviews, played_hours, playtime_last_two_weeks, votes_up, weighted_vote_score）生成：

- games_per_review_ratio = num_games_owned / (num_reviews + 1)
- hours_per_game_ratio = played_hours / (num_games_owned + 1)
- is_active_reviewer: num_reviews > 10 (0/1)
- is_hardcore_gamer: played_hours > 100 (0/1)
- is_casual_gamer: played_hours <= 10 (0/1)
- recent_activity_ratio = playtime_last_two_weeks / (playtime_at_review + 1)
- is_steam_purchaser = steam_purchase 转为 int
- received_free = received_for_free 转为 int
- author_reputation_score: 加权组合（log1p(num_games_owned) *0.2 + log1p(num_reviews)*0.3 + log1p(played_hours)*0.2 + optional votes_up、weighted_vote_score），随后做 z-score 标准化
- is_quality_reviewer, avg_votes_per_review, is_influential_reviewer: 基于 votes_up 和 num_reviews 的派生指标

注意：函数对数值列做数值化和缺失填充为 0。

## 评论互动与影响力特征（extract_review_engagement_features）

基于 votes_up, votes_funny, comment_count, weighted_vote_score, voted_up 等列计算：

- is_recommended: voted_up 转为 int
- votes_up_log, votes_funny_log, comment_count_log: log1p
- is_helpful_review: votes_up >=5
- is_highly_helpful: votes_up >=20
- is_funny_review: votes_funny >=3
- funny_ratio, serious_ratio: votes_funny / (votes_up+votes_funny+1), votes_up / total_votes
- is_controversial: comment_count >=10
- is_highly_discussed: comment_count >=30
- weighted_score_normalized: 直接使用 weighted_vote_score（并判定 is_high_quality if >=0.7）
- engagement_score: 0.5*votes_up_log + 0.2*votes_funny_log + 0.3*comment_count_log
- engagement_score_normalized: z-score 标准化（如果 std > 0）
- engagement_level: 四分位分段标签 ['low_engagement','medium_engagement','high_engagement','viral']
- negative_but_helpful: vader_compound < -0.3 & votes_up >= 10
- positive_but_unhelpful: vader_compound > 0.3 & votes_up <= 1
- controversial_sentiment: comment_count >=10 & abs(vader_compound) >=0.5
- recommendation_sentiment_mismatch: voted_up 与 vader_compound 的不一致情况

缺失值处理：数值列转为 0，ratio 类在分母加 1 以避免除零。

## 语法/句法特征（extract_syntax_features）

优先使用 spaCy（若可用）进行 POS 标注，并计算以下特征：

- prop_noun, prop_verb, prop_adj, prop_adv: 各 POS 在句子中的比例
- negation_count: 否定词计数
- intensifier_count: 强化/程度副词计数

如果 spaCy 不可用，则使用简单的启发式（仅计算 negation_count 与 intensifier_count，POS 比例置 0）。

## 交互与组级特征（extract_sentiment_interaction_features）

基于 group_key（优先 appid, app_id, genre）计算分组级情感统计：

- vader_compound_z: 在 group 内的 z-score
- sentiment_x_reputation: vader_compound * author_reputation_score（若存在）
- sentiment_x_bug: vader_compound * contains_bug_report（若存在）

## 输出与保存（save_features）

`save_features` 会将：
- 特征 DataFrame 保存为 CSV
- TF-IDF 矩阵保存为 .npz（scipy.sparse.save_npz）并写出特征词表
- BERT embeddings 保存为 .npy
- 生成并保存 feature summary 文本（包含分门别类的特征列表）
- 尝试保存 topic_pca 和 embedding_kmeans（若存在）为 pickle
