# FPS Risk 训练计划

## 📋 任务概述

**目标**: 训练风险评论识别模型（二分类），识别可能对游戏声誉/社区造成负面影响的高风险评论。

**数据**: FPS 游戏评论，176,848 样本
- Positive class (risk=1): 74,295 (42.01%) ✓ **相对平衡**
- Negative class (risk=0): 102,553 (57.99%)

**弱标签定义** (`risk_score >= 1.0`):
```python
risk_score = (
    is_toxic * 1.0 +                                    # 毒性内容
    (vader_compound < -0.35) * 1.0 +                    # 强负面情感
    contains_bug_report * 0.8 +                         # Bug 报告
    contains_balance_complaint * 0.8 +                  # 平衡性投诉
    contains_monetization_complaint * 0.8 +             # 付费投诉
    mentions_performance * 0.5 +                        # 性能问题
    controversial_sentiment * 0.5 +                     # 争议性情感
    recommendation_sentiment_mismatch * 0.6 +           # 推荐-情感不一致
    (negative_but_helpful + positive_but_unhelpful) * 0.3  # 矛盾反馈
)
```

**评估指标** (INTEGRATION_GUIDE 要求):
- **Primary**: **PR-AUC** (类别不平衡场景下的主要指标)
- **Secondary**: 
  - Recall@K (top 5%, 10%, 20%)
  - Precision, Recall, F1
  - False Positive Rate
  - Confusion Matrix

---

## 🎯 训练方案设计

### 方案对比：Baseline vs Advanced

| 维度 | Baseline | Advanced |
|------|----------|----------|
| **特征** | 仅文本 TF-IDF | 文本 + 表格特征融合 |
| **模型** | Logistic Regression | LightGBM / XGBoost |
| **训练时间** | ~5分钟 | ~20-30分钟 |
| **可解释性** | ✓✓✓ 高 | ✓✓ 中 |
| **预期性能** | PR-AUC 0.70-0.75 | PR-AUC 0.78-0.85 |
| **资源需求** | CPU | CPU (多核) |

---

## 📊 特征工程策略

### 基于今天的经验总结

#### ✅ **有效特征**（从 Sentiment/Topic 训练学到的）

1. **文本特征**（主特征）:
   - TF-IDF (10k features, trigram)
   - 已处理文本 `review_content_processed`
   - **经验**: Sentiment 训练中 TF-IDF Baseline 达到 82%+，证明文本特征强大

2. **情感特征**（次要但重要）:
   - `vader_compound`, `vader_neg`, `vader_pos`, `vader_neu`
   - `sentiment_score`, `controversial_sentiment`
   - `recommendation_sentiment_mismatch`
   - **经验**: 这些是 risk_score 的核心组成部分

3. **内容标记特征**:
   - `is_toxic`, `contains_bug_report`, `contains_balance_complaint`
   - `contains_monetization_complaint`, `mentions_performance`
   - **经验**: 直接对应风险定义，高权重

4. **异常检测特征**（新增价值）:
   - `ensemble_anomaly_score`, `ml_anomaly_score`, `statistical_anomaly_score`
   - **经验**: 异常评论往往是高风险评论

5. **作者行为特征**:
   - `author_playtime_at_review`, `author_num_games_owned`
   - `author_num_reviews`, `votes_helpful`, `votes_funny`
   - **经验**: 行为异常可能关联风险

#### ❌ **需要排除的特征**（避免泄露）

- `risk_score` (直接泄露目标)
- `risk_label_weak` (就是目标本身)
- `sentiment_label` (可能间接泄露，但可以作为特征测试)

---

## 🔧 实施细节

### 1. 数据准备

```python
# 加载数据
df = pd.read_parquet('gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet')

# 目标列
target = 'risk_label_weak'

# 排除列（避免泄露）
exclude_cols = [
    'risk_score',  # 直接泄露
    'risk_label_weak',  # 目标本身
    'risk_label',  # 如果存在
    'timestamp',  # 时间戳（用于排序，不作为特征）
    'review_datetime',  # 同上
    'review_id', 'author_id',  # ID 列
    # 文本原始列（使用处理后的）
    'review_content', 'review_content_clean'
]

# 时序切分（严格按时间）
df = df.sort_values('timestamp')
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# 或使用 TimeSeriesSplit (5-fold)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### 2. Baseline 模型（TF-IDF + LR）

**参考**: `train_sentiment_bert_fps.py` 的 Baseline 实现

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, 
    classification_report, roc_auc_score
)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(train_df['review_content_processed'])
X_test_vec = vectorizer.transform(test_df['review_content_processed'])

# Logistic Regression (class_weight='balanced' 处理不平衡)
clf = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='liblinear',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_vec, train_df[target])

# 预测概率（用于 PR-AUC）
y_proba = clf.predict_proba(X_test_vec)[:, 1]

# 计算 PR-AUC
precision, recall, _ = precision_recall_curve(test_df[target], y_proba)
pr_auc = auc(recall, precision)

print(f"Baseline PR-AUC: {pr_auc:.4f}")
```

**预期性能**:
- PR-AUC: 0.70-0.75
- F1: 0.65-0.70
- Recall@10%: 0.25-0.30

### 3. Advanced 模型（特征融合 + LightGBM）

**新增**: 融合文本 + 表格特征

```python
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 定义特征组
text_feature = 'review_content_processed'

numeric_features = [
    # 情感特征
    'vader_compound', 'vader_neg', 'vader_pos', 'vader_neu',
    'sentiment_score', 'controversial_sentiment',
    
    # 内容标记
    'is_toxic', 'contains_bug_report', 'contains_balance_complaint',
    'contains_monetization_complaint', 'mentions_performance',
    
    # 异常特征
    'ensemble_anomaly_score', 'ml_anomaly_score', 'statistical_anomaly_score',
    
    # 作者行为
    'author_playtime_at_review', 'author_num_games_owned',
    'author_num_reviews', 'votes_helpful', 'votes_funny',
    
    # 互动指标
    'votes_up', 'votes_down', 'comment_count'
]

# TF-IDF + 数值特征标准化
text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_feature),
        ('num', numeric_transformer, numeric_features)
    ])

# 训练 TF-IDF + 标准化
X_train_combined = preprocessor.fit_transform(train_df)
X_test_combined = preprocessor.transform(test_df)

# LightGBM (支持稀疏矩阵)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'scale_pos_weight': len(train_df[train_df[target]==0]) / len(train_df[train_df[target]==1])  # 处理不平衡
}

train_data = lgb.Dataset(X_train_combined, label=train_df[target])
test_data = lgb.Dataset(X_test_combined, label=test_df[target], reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 预测
y_proba_adv = model.predict(X_test_combined)

# 计算 PR-AUC
precision_adv, recall_adv, _ = precision_recall_curve(test_df[target], y_proba_adv)
pr_auc_adv = auc(recall_adv, precision_adv)

print(f"Advanced PR-AUC: {pr_auc_adv:.4f}")
```

**预期性能**:
- PR-AUC: 0.78-0.85
- F1: 0.72-0.78
- Recall@10%: 0.35-0.45

---

## 📈 评估指标实现

### Primary: PR-AUC

```python
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Method 1: 手动计算
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
pr_auc = auc(recall, precision)

# Method 2: 使用 average_precision_score (推荐)
pr_auc = average_precision_score(y_true, y_proba)

print(f"PR-AUC: {pr_auc:.4f}")
```

### Secondary: Recall@K

```python
def recall_at_k(y_true, y_proba, k=0.1):
    """
    计算 Recall@K (top k% 预测样本中的召回率)
    """
    n = len(y_true)
    top_k = int(n * k)
    
    # 按概率排序，取 top-k
    top_indices = np.argsort(y_proba)[-top_k:]
    
    # 计算 top-k 中的 TP
    tp = y_true.iloc[top_indices].sum()
    total_positives = y_true.sum()
    
    recall = tp / total_positives if total_positives > 0 else 0
    
    return recall

# 计算不同 k 值的 Recall
for k in [0.05, 0.10, 0.20]:
    recall_k = recall_at_k(test_df[target], y_proba, k=k)
    print(f"Recall@{k*100:.0f}%: {recall_k:.4f}")
```

### 完整评估报告

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 选择阈值（最大化 F1）
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

# 二分类预测
y_pred = (y_proba >= best_threshold).astype(int)

# 指标报告
print(f"\n{'='*80}")
print(f"EVALUATION REPORT (Threshold = {best_threshold:.4f})")
print(f"{'='*80}")

print(f"\nPrimary Metrics:")
print(f"  PR-AUC: {pr_auc:.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")

print(f"\nSecondary Metrics:")
print(classification_report(y_true, y_pred, target_names=['Low Risk', 'High Risk']))

print(f"\nRecall@K:")
for k in [0.05, 0.10, 0.20]:
    print(f"  Recall@{k*100:.0f}%: {recall_at_k(y_true, y_proba, k):.4f}")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

---

## 🔍 特征重要性分析（仅 Advanced）

```python
# LightGBM 特征重要性
feature_names = (
    [f'tfidf_{i}' for i in range(5000)] +  # TF-IDF 特征
    numeric_features  # 数值特征
)

importance = model.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Top 20 特征
print("\nTop 20 Most Important Features:")
print(feature_importance_df.head(20))

# 保存
feature_importance_df.to_csv('risk_feature_importance.csv', index=False)
```

---

## 🎨 可视化

### 1. PR Curve

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'Baseline (PR-AUC={pr_auc:.4f})')
plt.plot(recall_adv, precision_adv, label=f'Advanced (PR-AUC={pr_auc_adv:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve: Risk Classification')
plt.legend()
plt.grid(True)
plt.savefig('risk_pr_curve.png', dpi=300, bbox_inches='tight')
```

### 2. Score Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline
axes[0].hist(y_proba[y_true==0], bins=50, alpha=0.5, label='Risk=0')
axes[0].hist(y_proba[y_true==1], bins=50, alpha=0.5, label='Risk=1')
axes[0].set_title('Baseline Score Distribution')
axes[0].set_xlabel('Predicted Probability')
axes[0].legend()

# Advanced
axes[1].hist(y_proba_adv[y_true==0], bins=50, alpha=0.5, label='Risk=0')
axes[1].hist(y_proba_adv[y_true==1], bins=50, alpha=0.5, label='Risk=1')
axes[1].set_title('Advanced Score Distribution')
axes[1].set_xlabel('Predicted Probability')
axes[1].legend()

plt.tight_layout()
plt.savefig('risk_score_distribution.png', dpi=300, bbox_inches='tight')
```

### 3. Recall@K Comparison

```python
k_values = np.arange(0.01, 0.51, 0.01)
recall_baseline = [recall_at_k(y_true, y_proba, k) for k in k_values]
recall_advanced = [recall_at_k(y_true, y_proba_adv, k) for k in k_values]

plt.figure(figsize=(10, 6))
plt.plot(k_values * 100, recall_baseline, label='Baseline')
plt.plot(k_values * 100, recall_advanced, label='Advanced')
plt.xlabel('Top K%')
plt.ylabel('Recall')
plt.title('Recall@K Curve')
plt.legend()
plt.grid(True)
plt.savefig('risk_recall_at_k.png', dpi=300, bbox_inches='tight')
```

---

## 📝 输出文件结构

```
analysis_results/train/fps_risk/
├── baseline/
│   ├── model.pkl                          # Baseline 模型
│   ├── vectorizer.pkl                     # TF-IDF vectorizer
│   ├── metrics.json                       # 评估指标
│   ├── predictions.csv                    # 预测结果
│   └── confusion_matrix.png               # 混淆矩阵
├── advanced/
│   ├── model.txt                          # LightGBM 模型
│   ├── preprocessor.pkl                   # 特征预处理器
│   ├── metrics.json                       # 评估指标
│   ├── predictions.csv                    # 预测结果
│   ├── feature_importance.csv             # 特征重要性
│   └── feature_importance_top20.png       # Top 20 特征图
├── comparison/
│   ├── pr_curve.png                       # PR 曲线对比
│   ├── score_distribution.png             # 分数分布对比
│   ├── recall_at_k.png                    # Recall@K 对比
│   └── comparison_report.json             # 对比总结
└── risk_training_report.md                # 完整训练报告
```

---

## ⏱️ 时间规划

| 步骤 | 预计时间 | 说明 |
|------|---------|------|
| 数据加载与预处理 | 5分钟 | 排除泄露列，时序切分 |
| Baseline 训练 | 5分钟 | TF-IDF + LR，5折CV |
| Baseline 评估 | 2分钟 | PR-AUC, Recall@K |
| Advanced 特征工程 | 10分钟 | 文本+表格特征融合 |
| Advanced 训练 | 20-30分钟 | LightGBM, 5折CV |
| Advanced 评估 | 5分钟 | 完整指标 + 特征重要性 |
| 可视化与报告 | 10分钟 | 生成图表和总结 |
| **总计** | **~60分钟** | 单 genre (FPS) |

**Per-genre 扩展**: Leisure + Strategy 各需额外 30-45 分钟。

---

## 🚀 实施优先级

### Phase 1: FPS Risk (立即执行)
- ✅ 数据已准备（176k样本，42% positive）
- ✅ 所有特征可用（100% coverage）
- 📝 创建 `train_risk_fps.py` (基于 `train_sentiment_bert_fps.py` 模板)
- 🎯 目标: Baseline PR-AUC 0.70+, Advanced PR-AUC 0.78+

### Phase 2: Per-Genre Risk (可选)
- Leisure Risk 训练
- Strategy Risk 训练
- 对比 FPS vs Leisure vs Strategy 的风险模式差异

### Phase 3: Risk 告警系统集成
- 基于 Advanced 模型预测概率
- 设置告警阈值（如 top 10% 为高风险）
- 集成到综合告警系统

---

## 🔧 关键经验应用（从 Sentiment/Topic 学到的）

### ✅ **有效实践**

1. **时序切分严格性**
   - 经验: Sentiment 训练中严格时序切分避免了数据泄露
   - 应用: Risk 训练同样使用 `TimeSeriesSplit`

2. **类别权重平衡**
   - 经验: Sentiment 使用 `class_weight='balanced'` 处理不平衡
   - 应用: Risk 的 42% positive 相对平衡，但仍建议使用 `scale_pos_weight`

3. **TF-IDF 参数优化**
   - 经验: `max_features=10000, ngram_range=(1,3)` 在 Sentiment 中效果好
   - 应用: Risk 文本特征同样使用这些参数

4. **PR-AUC 作为主指标**
   - 经验: Topic 训练中 Weighted F1 适合不平衡场景
   - 应用: Risk 使用 PR-AUC（INTEGRATION_GUIDE 明确要求）

5. **5折 CV 估计稳定性**
   - 经验: Sentiment/Topic 都使用 5折 CV
   - 应用: Risk 同样使用 5折 TimeSeriesSplit

### ⚠️ **需要注意的问题**

1. **特征泄露风险**
   - 问题: `risk_score` 直接包含目标信息
   - 解决: 严格排除 `risk_score` 及相关列

2. **Recall@K 的业务意义**
   - 问题: 高风险评论需要优先处理
   - 解决: 重点关注 Recall@10% (top 10% 预测中的召回率)

3. **阈值选择策略**
   - 问题: 不同阈值对应不同的 Precision-Recall 权衡
   - 解决: 提供阈值曲线，让业务方根据成本选择

---

## 📊 预期成果

### Baseline (TF-IDF + LR)
- PR-AUC: **0.70-0.75**
- Recall@10%: **0.25-0.30**
- F1: **0.65-0.70**
- 训练时间: **~5分钟**

### Advanced (Features + LightGBM)
- PR-AUC: **0.78-0.85** (↑ 8-10%)
- Recall@10%: **0.35-0.45** (↑ 50%+)
- F1: **0.72-0.78** (↑ 7-8%)
- 训练时间: **~30分钟**

### 关键洞察
- 文本特征主导（TF-IDF 贡献 ~70%）
- 异常检测特征增强识别能力（贡献 ~15%）
- 情感特征提供补充信息（贡献 ~15%）

---

## 📌 TODO List

- [ ] 创建 `train_risk_fps.py` (基于 Sentiment 模板)
- [ ] 实现 Baseline 模型 (TF-IDF + LR)
- [ ] 实现 Advanced 模型 (Features + LightGBM)
- [ ] 实现完整评估指标 (PR-AUC, Recall@K, etc.)
- [ ] 生成可视化图表 (PR Curve, Score Distribution, etc.)
- [ ] 创建训练报告模板
- [ ] 运行完整训练（5折CV，60分钟）
- [ ] 对比 Baseline vs Advanced
- [ ] 集成到告警系统

---

**文档版本**: v1.0  
**创建日期**: 2025-10-25  
**参考**: INTEGRATION_GUIDE.md, train_sentiment_bert_fps.py, train_topic_fps.py
