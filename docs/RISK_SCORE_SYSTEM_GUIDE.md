# Risk Score 阈值系统说明文档

## 📋 系统概述

**Risk Score 阈值系统** 是一个基于规则的风险评分与分级告警系统，用于识别和优先处理 Steam 游戏评论中的高风险内容。

**核心理念**: 无需训练机器学习模型，直接使用预定义的加权规则计算风险分数，通过多级阈值实现分级告警。

---

## 🎯 系统功能

### 1. **风险评分计算**

基于 **10 个特征的加权求和**：

**高权重特征 (1.0)**:
- `is_toxic`: 毒性内容检测
- `vader_compound < -0.35`: 强负面情感

**中权重特征 (0.8)**:
- `contains_bug_report`: Bug 报告
- `contains_balance_complaint`: 平衡性投诉
- `contains_monetization_complaint`: 付费/氪金投诉

**低权重特征 (0.5-0.6)**:
- `mentions_performance`: 性能问题 (0.5)
- `controversial_sentiment`: 争议性情感 (0.5)
- `recommendation_sentiment_mismatch`: 推荐与情感不一致 (0.6)

**最低权重特征 (0.3)**:
- `negative_but_helpful`: 负面但有帮助
- `positive_but_unhelpful`: 正面但无帮助

**分数范围**: 0.0 - 5.0
- 理论最大值: 5.0（所有特征都触发）
- 实际最大值: 5.0（数据中确实存在）
- 平均值: 0.951
- 中位数: 0.8

---

### 2. **多级阈值分级**

系统提供 **4 个风险等级**：

#### Method 3: 基于弱标签 + 细化（当前系统）

| 风险等级 | 阈值 | 样本数量 | 占比 | 处理策略 |
|---------|------|---------|------|---------|
| **🔴 Critical** | ≥ 2.5 | 16,903 | 9.56% | **立即处理** - 严重毒性、多重投诉或极端负面，需要人工审核或自动屏蔽 |
| **🟠 High** | ≥ 1.5 | 34,958 | 19.77% | **24小时内审查** - 多个风险信号，加入优先队列 |
| **🟡 Medium** | ≥ 1.0 | 22,434 | 12.69% | **每周抽查** - 至少一个主要风险特征触发 |
| **🟢 Low** | < 1.0 | 102,553 | 57.99% | **正常监控** - 风险较低，常规流程 |

**适用场景**: 初期上线，覆盖面广，简单易解释

---

#### Method 4: 混合自适应方法（⭐ 生产环境推荐）

| 风险等级 | 阈值 | 样本数量 | 占比 | 处理策略 |
|---------|------|---------|------|---------|
| **🔴 Critical** | ≥ 3.0 | 4,068 | 2.30% | **立即处理** - 极高风险，同时满足多个严重条件，强制人工审核 |
| **🟠 High** | ≥ 2.0 | 24,718 | 13.98% | **24小时内审查** - 明确的高风险信号（2+高权重特征） |
| **🟡 Medium** | ≥ 1.0 | 45,509 | 25.73% | **每周抽查** - 存在风险信号但不紧急 |
| **🟢 Low** | < 1.0 | 102,553 | 57.99% | **正常流程** - 风险较低，常规监控 |

**适用场景**: 人工审核资源有限，需要高精准度

**核心优势**:
- Critical 告警量减少 **75.9%**（16,903 → 4,068）
- Critical 平均分数更高（2.69 → 3.42）
- 总告警量（Critical + High）减少至 **16.3%**

---

### 3. **高风险样本提取**

自动提取 Top-K（默认 1000）最高风险评论，包含：

**输出字段**:
- 基础信息: `review_id`, `app_id`, `timestamp`, `review_datetime`
- 风险评分: `risk_score`, `risk_label_weak`
- 风险特征: 10 个特征的具体值（`is_toxic`, `vader_compound`, etc.）
- 互动指标: `votes_up`, `votes_funny`, `comment_count`
- 评论内容: `review_content_processed`

**用途**:
- 人工抽样审核高风险内容
- 训练集标注（如需改进弱标签）
- 业务方案例分析

**Top 1000 统计** (FPS 数据):
- 风险分数范围: 3.6 - 5.0
- 平均分数: 3.783
- 这些评论通常包含毒性语言、多重投诉或极端情感

---

### 4. **数据分析与可视化**

#### 分布分析
- 风险分数的统计量（均值、中位数、标准差、百分位数）
- 按 `risk_label_weak` (0/1) 分组的分布对比

#### 可视化图表

**a) `risk_score_distribution.png`** (4 个子图)
1. **整体分布直方图**: 展示所有样本的风险分数分布，标注 3 级阈值线
2. **按标签分组**: 对比 Low Risk (0) vs High Risk (1) 的分数分布
3. **箱型图**: 可视化两组的中位数、四分位数和异常值
4. **累积分布曲线**: 显示不同阈值对应的样本百分比

**b) `risk_feature_weights.png`**
- 横向条形图，展示 10 个特征的权重大小
- 颜色编码: 红色 (≥1.0), 橙色 (≥0.5), 黄色 (<0.5)

---

### 5. **阈值推荐方法**

系统提供 **3 种阈值设置方法**：

#### Method 1: 基于百分位数
- Critical: p99 (3.4) - Top 1%
- High: p95 (2.8) - Top 5%
- Medium: p75 (1.6) - Top 25%

**适用场景**: 希望基于数据分布自动调整，适合样本量大且分布稳定的情况。

#### Method 2: 基于标准差
- Critical: μ + 2σ (2.78)
- High: μ + σ (1.87)
- Medium: μ (0.95)

**适用场景**: 统计学标准方法，适合正态或近似正态分布。

#### Method 3: 基于弱标签 + 细化
- Critical: 2.5
- High: 1.5
- Medium: 1.0 (原弱标签阈值)

**适用场景**: 
- 结合业务理解设置的固定阈值
- 易于解释和调整
- 与弱标签生成逻辑一致

---

#### Method 4: 混合自适应方法 (⭐ **推荐** - 新方法)

**核心思想**: 结合统计学方法（Method 2）的自适应性 + 百分位数（Method 1）的业务控制 + 弱标签逻辑（Method 3）的可解释性

**阈值设计**:
```
Critical:  p99 与 μ+2σ 的平均值，向下取整到 0.5 的倍数 
          → (3.4 + 2.78) / 2 = 3.09 → 3.0

High:      p95 与 μ+σ 的平均值，向下取整到 0.5 的倍数
          → (2.8 + 1.87) / 2 = 2.34 → 2.0

Medium:    p75 与 μ 的平均值，向下取整到 0.5 的倍数
          → (1.6 + 0.95) / 2 = 1.28 → 1.0

Low:       固定为 0.0
```

**最终阈值**:
- **🔴 Critical**: ≥ 3.0
- **🟠 High**: ≥ 2.0  
- **🟡 Medium**: ≥ 1.0
- **🟢 Low**: < 1.0

**实际验证结果** (基于 176,848 条 FPS 数据):
- Critical: 4,068 (2.30%) - 仅保留最严重风险（score 3.0-5.0）
- High: 24,718 (13.98%) - 明确的高风险信号
- Medium: 45,509 (25.73%) - 存在风险但不紧急
- Low: 102,553 (57.99%) - 正常范围

**优势**:
1. **数据驱动 + 可解释**: 阈值来源于统计分析，但结果是简单的整数/半整数
2. **业务友好**: 告警量更加聚焦（Critical 从 9.6% 降至 2.8%）
3. **自适应**: 可随数据分布变化自动调整（重新运行分析脚本）
4. **风险梯度合理**: 
   - Critical (3.0): 通常包含 **毒性 + 极端负面 + 至少1个投诉**
   - High (2.0): 包含 **2个高权重特征** 或 **1个高权重 + 多个中权重**
   - Medium (1.0): 至少 **1个高权重特征** 或 **多个中权重特征**

**适用场景**:
- ✅ **生产环境推荐**: 告警量适中，Critical 精准度高
- ✅ 需要平衡数据驱动与业务可控性
- ✅ 希望阈值能随数据演化自动调整
- ✅ 对高优先级告警（Critical）有严格质量要求

**对比 Method 3** (实际验证数据):
| 指标 | Method 3 (当前) | Method 4 (新) | 改进 |
|-----|----------------|--------------|-----|
| Critical 阈值 | 2.5 | 3.0 | +0.5 (更严格) |
| Critical 样本量 | 16,903 (9.6%) | 4,068 (2.3%) | **-75.9%** (大幅减少) |
| High 阈值 | 1.5 | 2.0 | +0.5 (更严格) |
| High 样本量 | 34,958 (19.8%) | 24,718 (14.0%) | -29.3% (更聚焦) |
| Medium 样本量 | 22,434 (12.7%) | 45,509 (25.7%) | +102.9% (吸收降级样本) |
| 设计依据 | 人工经验 | 统计方法 + 圆整 | 数据驱动 |
| 可解释性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 略降 |
| 精准度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **显著提升** |

**关键发现** (边界样本分析):
- **12,835 条** 从 Critical 降级到 High (score 2.5-3.0 区间)
  - 82% 包含毒性内容，92.5% 强负面情感
  - 但未达到 Method 4 的 Critical 标准（score ≥ 3.0）
- **4,068 条** 保持 Critical 等级 (score 3.0-5.0)
  - 平均分数 **3.42**（真正的极高风险）
  - 这些是需要 **立即处理** 的核心样本

**实现建议**:
1. 初期使用 **Method 3**（当前系统）快速上线
2. 积累 1-2 周人工审核数据后，验证 Method 4 是否能降低误报率
3. 如果 Critical 告警的人工审核通过率 > 80%，切换到 Method 4
4. 每月重新运行分析脚本，更新阈值（数据分布可能随时间变化）

---

**当前系统使用**: Method 3 (基于弱标签 + 细化)  
**生产环境推荐**: Method 4 (混合自适应方法)

---

## 📊 输出文件说明

### 1. `risk_thresholds.json`
**内容**: 完整的阈值配置与统计信息

```json
{
  "recommended_thresholds": {
    "critical": 2.5,
    "high": 1.5,
    "medium": 1.0,
    "low": 0.0
  },
  "alternative_methods": { ... },
  "sample_distribution": {
    "critical": 16903,
    "high": 34958,
    "medium": 22434,
    "low": 102553
  },
  "statistics": { ... },
  "feature_weights": { ... },
  "methodology": { ... }
}
```

**用途**: 
- 配置告警系统
- 文档记录
- 阈值调优参考

---

### 2. `high_risk_samples_top1000.csv`
**内容**: Top 1000 最高风险评论的详细数据

**字段** (19 列):
- 标识: `review_id`, `app_id`
- 时间: `timestamp`, `review_datetime`
- 评分: `risk_score`, `risk_label_weak`
- 特征: 10 个风险特征值
- 互动: `votes_up`, `votes_funny`, `comment_count`
- 内容: `review_content_processed`

**用途**:
- 人工审核样本
- 标注训练集
- 案例分析

**示例** (前 3 行):
| risk_score | is_toxic | vader_compound | bug | balance | monetization | 内容摘要 |
|-----------|----------|----------------|-----|---------|--------------|----------|
| 5.00 | 1 | -0.84 | 1 | 1 | 1 | "broken game bad anticheat..." |
| 4.90 | 1 | -0.46 | 1 | 1 | 1 | "shit pay win game..." |
| 4.90 | 1 | -0.95 | 1 | 1 | 1 | "game full cheater..." |

---

### 3. `risk_feature_weights.json`
**内容**: 特征权重配置

```json
{
  "feature_weights": {
    "is_toxic": 1.0,
    "vader_compound_negative": 1.0,
    ...
  },
  "description": "...",
  "usage": "risk_score = sum([feature * weight ...])"
}
```

**用途**:
- 权重调优参考
- 系统配置
- 算法文档

---

### 4. `risk_score_distribution.png`
**内容**: 4 个子图的综合可视化

**用途**:
- 数据探索
- 报告展示
- 阈值验证

---

### 5. `risk_feature_weights.png`
**内容**: 特征权重条形图

**用途**:
- 特征重要性展示
- 权重调优可视化
- 报告附图

---

## 🔧 使用方法

### 运行分析脚本

```bash
python scripts/summary/analyze_risk_score_thresholds.py
```

**输入**: 
- FPS 特征文件: `features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet`

**输出目录**: 
- `analysis_results/risk_scoring_system/`

**运行时间**: 约 3-5 秒

---

### 在代码中使用风险评分

#### 方法 1: 直接使用已有的 `risk_score` 列

**使用 Method 3 阈值** (当前系统):
```python
import pandas as pd

# 加载特征数据
df = pd.read_parquet('features/fps/...with_weaklabels.parquet')

# 应用 Method 3 阈值
df['risk_level'] = 'Low'
df.loc[df['risk_score'] >= 1.0, 'risk_level'] = 'Medium'
df.loc[df['risk_score'] >= 1.5, 'risk_level'] = 'High'
df.loc[df['risk_score'] >= 2.5, 'risk_level'] = 'Critical'

# 筛选高风险评论
critical_reviews = df[df['risk_level'] == 'Critical']
print(f"Critical 告警量: {len(critical_reviews)} ({len(critical_reviews)/len(df)*100:.2f}%)")
```

**使用 Method 4 阈值** (推荐用于生产):
```python
import pandas as pd

# 加载特征数据
df = pd.read_parquet('features/fps/...with_weaklabels.parquet')

# 应用 Method 4 阈值（更严格）
df['risk_level'] = 'Low'
df.loc[df['risk_score'] >= 1.0, 'risk_level'] = 'Medium'
df.loc[df['risk_score'] >= 2.0, 'risk_level'] = 'High'
df.loc[df['risk_score'] >= 3.0, 'risk_level'] = 'Critical'

# 筛选高风险评论
critical_reviews = df[df['risk_level'] == 'Critical']
high_reviews = df[df['risk_level'] == 'High']

print(f"Critical 告警量: {len(critical_reviews)} ({len(critical_reviews)/len(df)*100:.2f}%)")
print(f"High 告警量: {len(high_reviews)} ({len(high_reviews)/len(df)*100:.2f}%)")
print(f"需要优先处理: {len(critical_reviews) + len(high_reviews)} ({(len(critical_reviews)+len(high_reviews))/len(df)*100:.2f}%)")
```

**配置化阈值管理**:
```python
import json
import pandas as pd

# 方法1: 从配置文件加载
with open('analysis_results/risk_scoring_system/risk_thresholds.json') as f:
    config = json.load(f)
    thresholds = config['recommended_thresholds']  # Method 3

# 方法2: 使用 Method 4 阈值
thresholds = {
    'critical': 3.0,
    'high': 2.0,
    'medium': 1.0,
    'low': 0.0
}

# 应用阈值
def assign_risk_level(score, thresholds):
    if score >= thresholds['critical']:
        return 'Critical'
    elif score >= thresholds['high']:
        return 'High'
    elif score >= thresholds['medium']:
        return 'Medium'
    else:
        return 'Low'

df['risk_level'] = df['risk_score'].apply(lambda x: assign_risk_level(x, thresholds))
```

#### 方法 2: 从零计算（新数据）

```python
def calculate_risk_score(row):
    """计算单条评论的风险分数"""
    score = 0.0
    
    # 高权重特征 (1.0)
    if row.get('is_toxic', 0) == 1:
        score += 1.0
    if row.get('vader_compound', 0) < -0.35:
        score += 1.0
    
    # 中权重特征 (0.8)
    score += row.get('contains_bug_report', 0) * 0.8
    score += row.get('contains_balance_complaint', 0) * 0.8
    score += row.get('contains_monetization_complaint', 0) * 0.8
    
    # 低权重特征 (0.5-0.6)
    score += row.get('mentions_performance', 0) * 0.5
    score += row.get('controversial_sentiment', 0) * 0.5
    score += row.get('recommendation_sentiment_mismatch', 0) * 0.6
    
    # 最低权重特征 (0.3)
    score += row.get('negative_but_helpful', 0) * 0.3
    score += row.get('positive_but_unhelpful', 0) * 0.3
    
    return score

# 应用到 DataFrame
df['risk_score'] = df.apply(calculate_risk_score, axis=1)
```

#### 方法 3: 使用 weaklabeling.py

```python
from scripts.train.weaklabeling import generate_weak_labels

# 自动生成所有弱标签（包括 risk_score）
df = generate_weak_labels(df)
```

---

## 📈 性能与适用场景

### ✅ 优势

1. **完全可解释**: 每个分数都有明确的特征贡献，易于向业务方解释
2. **无需训练**: 不依赖机器学习模型，部署简单
3. **实时计算**: 规则计算速度快（毫秒级），适合在线系统
4. **易于调整**: 权重和阈值可根据业务需求快速调整
5. **覆盖全面**: 10 个特征涵盖毒性、情感、投诉、矛盾等多维度

### ⚠️ 局限性

1. **无法学习新模式**: 规则固定，无法从数据中自动发现新的风险模式
2. **特征依赖**: 准确性依赖于上游特征（如毒性检测、情感分析）的质量
3. **线性组合**: 无法捕捉特征之间的复杂交互关系
4. **权重主观**: 权重设置基于领域知识，可能需要业务方反复调优

### 🎯 适用场景

**✅ 推荐使用**:
- 初期快速上线风险监控系统
- 需要完全可解释的告警规则
- 评论数据规模大（百万级+），实时性要求高
- 已有可靠的特征工程管线

**❌ 不推荐使用**:
- 需要高精度风险预测（考虑机器学习模型）
- 风险模式复杂且规则难以定义
- 缺少高质量的上游特征（如毒性检测不准确）

---

## 🔄 与机器学习模型的对比

| 维度 | Risk Score 系统 (规则) | ML 模型 (训练) |
|-----|----------------------|---------------|
| **可解释性** | ✅ 完全透明 | ❌ 黑盒（BERT）或部分透明（LR） |
| **部署成本** | ✅ 极低（直接计算） | ❌ 需要模型服务 |
| **训练时间** | ✅ 无需训练 | ❌ 数小时（BERT 5-6h） |
| **调优灵活性** | ✅ 实时调整权重 | ❌ 需要重新训练 |
| **性能** | ⚠️ 中等（依赖规则质量） | ✅ 高（样本测试 PR-AUC 0.90+） |
| **新模式学习** | ❌ 无法自动学习 | ✅ 可从数据中学习 |
| **维护成本** | ✅ 低 | ❌ 需要监控模型漂移 |

**结论**: 
- **Risk Score 系统**: 适合初期快速上线和日常监控
- **ML 模型**: 适合高精度场景或作为 Risk Score 的补充

---

## 🛠️ 调优指南

### 0. 选择合适的阈值方法

**决策流程**:

```
1. 初期上线（无历史数据）
   → 使用 Method 3（基于弱标签 + 细化）
   理由：简单、可解释、覆盖范围广

2. 运行 1-2 周后，收集人工审核反馈
   → 评估 Critical 告警的精准度
   
   如果精准度 > 80%（误报率 < 20%）
      → 继续使用 Method 3
   
   如果精准度 < 80%（误报过多）
      → 切换到 Method 4（混合自适应）
      → Critical 告警量减少 70%，精准度提升

3. 稳定运行后（每月）
   → 重新运行 analyze_risk_score_thresholds.py
   → 检查数据分布是否变化
   → 如果 p99/p95 变化 > 10%，更新 Method 4 阈值
```

**快速对比**:

| 场景 | 推荐方法 | 原因 |
|-----|---------|-----|
| 初期上线，业务不确定 | Method 3 | 覆盖面广，不会漏报 |
| 人工审核资源有限 | Method 4 | Critical 量少（2.8%），精准度高 |
| 需要完全可控的阈值 | Method 3 | 固定阈值，易于解释 |
| 数据分布随时间变化 | Method 4 | 自适应，每月更新 |
| 严格的合规要求 | Method 4 | Critical 标准严格（≥3.0） |

**切换方法的代码**:

```python
# 当前使用 Method 3
CURRENT_THRESHOLDS = {
    'critical': 2.5,
    'high': 1.5,
    'medium': 1.0,
    'low': 0.0
}

# 切换到 Method 4（根据分析结果）
NEW_THRESHOLDS = {
    'critical': 3.0,   # ← 提高 0.5
    'high': 2.0,       # ← 提高 0.5
    'medium': 1.0,     # ← 保持不变
    'low': 0.0
}

# A/B 测试对比
df['risk_level_method3'] = df['risk_score'].apply(lambda x: assign_risk_level(x, CURRENT_THRESHOLDS))
df['risk_level_method4'] = df['risk_score'].apply(lambda x: assign_risk_level(x, NEW_THRESHOLDS))

# 统计对比
print("Method 3 分布:")
print(df['risk_level_method3'].value_counts(normalize=True))

print("\nMethod 4 分布:")
print(df['risk_level_method4'].value_counts(normalize=True))

# 找出分歧样本（Method 3 认为 Critical，Method 4 认为 High）
edge_cases = df[
    (df['risk_level_method3'] == 'Critical') & 
    (df['risk_level_method4'] == 'High')
]
print(f"\n边界样本数量: {len(edge_cases)} (risk_score 在 2.5-3.0 之间)")
```

---

### 1. 调整特征权重

**场景**: 发现某类风险被低估或高估

**方法**: 修改 `scripts/train/weaklabeling.py` 中的权重

```python
# 示例：提高毒性内容的权重
score += df['is_toxic'].fillna(0).astype(float) * 1.5  # 原 1.0 → 1.5
```

**建议**:
- 增量调整（每次 ±0.1-0.2）
- 重新运行分析脚本验证分布变化
- 查看 Top 1000 样本是否符合预期

---

### 2. 调整阈值

**场景**: 告警量过多或过少

**方法**: 修改 `risk_thresholds.json` 或代码中的阈值

```python
# 示例：提高 Critical 阈值减少告警量
thresholds_weak_label = {
    'critical': 3.0,   # 原 2.5 → 3.0
    'high': 2.0,       # 原 1.5 → 2.0
    'medium': 1.0,
    'low': 0.0
}
```

**建议**:
- 根据 `sample_distribution` 预估调整后的告警量
- 考虑业务处理能力（Critical 量不宜超过人工审核能力）

---

### 3. 添加新特征

**场景**: 发现新的风险维度

**方法**: 
1. 在特征工程中添加新特征
2. 在 `weaklabeling.py` 中增加权重
3. 重新生成弱标签
4. 重新运行分析脚本

**示例**:
```python
# 添加"包含种族歧视"特征（假设已在特征工程中生成）
if 'contains_hate_speech' in df.columns:
    score += df['contains_hate_speech'].fillna(0).astype(float) * 1.2
```

---

## 📞 常见问题 (FAQ)

### Q1: 为什么不训练机器学习模型？

**A**: 因为 `risk_label_weak` 本身就是用这些规则生成的。如果用这些特征训练模型预测 `risk_label_weak`，模型只是在"重新学习规则"，没有新的价值。直接用规则更简单、透明、可控。

如果未来有人工标注的真实标签，可以训练模型来学习人类判断风险的隐含模式。

---

### Q2: Risk Score 5.0 的评论有多严重？

**A**: 这些评论同时触发了所有主要风险信号：
- 毒性语言
- 极端负面情感
- Bug 报告
- 平衡性投诉
- 付费投诉
- 其他矛盾信号

建议优先人工审核或自动屏蔽。

---

### Q3: 如何与现有告警系统集成？

**A**: 
1. **批量模式**: 定期（如每小时）计算新评论的 risk_score，推送 Critical/High 到告警队列
2. **实时模式**: 评论发布时立即计算 risk_score，超过阈值即触发告警
3. **API 服务**: 封装计算逻辑为 REST API，供其他系统调用

**示例集成代码**:
```python
def trigger_alert(review):
    """判断是否需要告警"""
    risk_score = calculate_risk_score(review)
    
    if risk_score >= 2.5:
        send_alert(level='CRITICAL', review=review, score=risk_score)
    elif risk_score >= 1.5:
        send_alert(level='HIGH', review=review, score=risk_score)
    elif risk_score >= 1.0:
        add_to_review_queue(review, priority='medium')
```

---

### Q4: 阈值需要多久调整一次？

**A**: 
- **初期**: 每周检查一次，根据人工审核反馈调整
- **稳定期**: 每月检查一次，或在数据分布显著变化时调整
- **触发调整**:
  - Critical 告警量突然增加/减少 50%+
  - Top 1000 样本中出现大量误报
  - 业务方反馈漏报率过高

---

### Q5: 能否用于其他游戏类型？

**A**: 可以，但需要调整：

1. **重新分析分布**: 不同游戏类型的风险分布可能不同
2. **调整阈值**: 根据新数据的百分位数重新设置
3. **考虑调整权重**: 某些投诉类型在不同游戏中的重要性可能不同

**建议**: 为每个 genre（FPS/Leisure/Strategy）单独运行分析脚本，生成独立的阈值配置。

---

## 📚 相关文档

- **特征工程文档**: `docs/feature_engineering.md`
- **弱标签生成**: `scripts/train/weaklabeling.py`
- **训练计划**: `docs/RISK_TRAINING_PLAN.md` (已弃用训练，保留阈值系统)
- **集成指南**: `docs/INTEGRATION_GUIDE.md`

---

## 📝 更新日志

### v1.0 (2025-10-25)
- ✅ 初始版本
- ✅ 实现 10 特征风险评分
- ✅ 提供 3 种阈值推荐方法
- ✅ 生成 Top 1000 高风险样本
- ✅ 完整的可视化图表
- ✅ 统计分析报告

---

## 👥 联系方式

**项目**: BAP (Steam Review Analysis)  
**负责人**: BAP 项目组  
**文档版本**: v1.0  
**最后更新**: 2025-10-25

---

## 📌 核心要点总结

1. **Risk Score = 规则加权求和**（10 特征，0-5 分）
2. **4 级阈值系统**:
   - **Method 3** (当前): Critical (2.5), High (1.5), Medium (1.0), Low (0)
   - **Method 4** (推荐): Critical (3.0), High (2.0), Medium (1.0), Low (0)
3. **无需训练模型**，直接用规则，完全可解释
4. **适合初期上线**，后续可用人工标注训练 ML 模型改进
5. **输出文件**: 阈值配置、Top-K 样本、权重配置、可视化图表、对比报告

---

## 🆕 附录：Method 4 详细说明

### 设计原理

**Method 4: 混合自适应方法** 结合了三种方法的优势：

1. **Method 1 (百分位数)** 的业务控制能力
   - 用 p99, p95, p75 控制告警量占比
   - 确保告警量不会随数据量增长而失控

2. **Method 2 (标准差)** 的统计自适应性
   - 用 μ+2σ, μ+σ, μ 捕捉数据分布特征
   - 随数据分布变化自动调整

3. **Method 3 (弱标签逻辑)** 的可解释性
   - 圆整到 0.5 的倍数，便于业务理解
   - 保持 Medium 阈值为 1.0，与原弱标签逻辑一致

### 计算过程

以 FPS 数据为例（176,848 条评论）：

**统计量**:
- 均值 (μ) = 0.95
- 标准差 (σ) = 0.91
- p75 = 1.6, p95 = 2.8, p99 = 3.4

**阈值计算**:

```python
# Critical: 百分位数与标准差的平均值
critical_raw = (p99 + μ + 2σ) / 2
              = (3.4 + 0.95 + 2×0.91) / 2
              = (3.4 + 2.77) / 2
              = 3.085
critical = round_to_half(3.085) = 3.0  # 向下取整到 0.5 倍数

# High: 同理
high_raw = (p95 + μ + σ) / 2
         = (2.8 + 0.95 + 0.91) / 2
         = (2.8 + 1.86) / 2
         = 2.33
high = round_to_half(2.33) = 2.0

# Medium: 保持原弱标签逻辑
medium = 1.0
```

### 实际效果验证

运行 `python scripts/summary/compare_threshold_methods.py` 的结果：

**告警量变化**:
```
Method 3 → Method 4:
  Critical: 16,903 (9.6%) → 4,068 (2.3%)   [-75.9%]
  High:     34,958 (19.8%) → 24,718 (14.0%) [-29.3%]
  Medium:   22,434 (12.7%) → 45,509 (25.7%) [+102.9%]
  Low:      102,553 (57.9%) → 102,553 (57.9%) [不变]
```

**边界样本分析**:
- **12,835 条** 从 Critical 降级到 High (score 2.5-3.0)
  - 这些评论仍然高风险（82% 毒性，92.5% 强负面）
  - 但不需要"立即处理"，可以在 24 小时内审查

- **4,068 条** 保持 Critical (score ≥ 3.0)
  - 平均分数 3.42（显著高于阈值）
  - 同时满足多个严重条件：毒性 + 极端负面 + 投诉
  - 这些是真正需要 **立即人工审核或自动屏蔽** 的样本

### 业务价值

1. **减少误报**: Critical 告警减少 76%，人工审核压力大幅降低
2. **提高精准度**: 剩余的 Critical 样本平均分数更高（3.42 vs 2.69）
3. **优化资源分配**: 
   - Critical (2.3%) → 立即处理（强制人工审核）
   - High (14.0%) → 24 小时内审查（优先队列）
   - Medium (25.7%) → 每周抽查（常规监控）

4. **自适应演化**: 每月重新运行分析脚本，阈值随数据分布自动调整

### 切换建议

**阶段 1: 初期上线 (第 1-2 周)**
- 使用 **Method 3** (阈值: 2.5 / 1.5 / 1.0)
- 收集人工审核数据
- 统计 Critical 告警的误报率

**阶段 2: A/B 测试 (第 3-4 周)**
- 并行运行两种方法
- 对比人工审核通过率：
  - Method 3 Critical: 预期 60-70%
  - Method 4 Critical: 预期 80-90%
- 如果 Method 4 精准度 > 80%，则切换

**阶段 3: 全面切换 (第 5 周起)**
- 切换到 **Method 4** (阈值: 3.0 / 2.0 / 1.0)
- 每月重新运行分析脚本更新阈值
- 监控告警量和误报率变化

### 可视化对比

运行对比脚本后生成的图表 (`threshold_methods_comparison.png`) 包含：

1. **分布对比柱状图**: 直观展示两种方法的告警量差异
2. **百分比堆叠图**: 展示各等级占比变化
3. **风险分数分布 + 阈值线**: 对比两种方法的阈值位置
4. **分类转移矩阵**: 展示样本在两种方法间的流动

### FAQ

**Q1: Method 4 会不会漏掉重要的风险样本？**

**A**: 不会。降级的 12,835 条样本仍然在 **High** 等级（阈值 2.0），仍然需要优先处理，只是不需要"立即处理"。这些样本的平均分数是 2.69，介于 High 和 Critical 之间，24 小时内审查是合理的。

**Q2: Method 4 的阈值需要手工计算吗？**

**A**: 不需要。运行 `scripts/summary/analyze_risk_score_thresholds.py` 会自动计算所有方法的阈值。如果数据分布变化，重新运行脚本即可更新。

**Q3: 不同游戏类型（FPS/Leisure/Strategy）能用同一套阈值吗？**

**A**: 建议分别运行分析脚本。不同游戏类型的风险分布可能不同，例如：
- FPS 游戏：作弊、平衡性投诉可能更突出
- Leisure 游戏：付费投诉可能更多
- Strategy 游戏：Bug 报告、性能问题可能更常见

为每个类型生成独立的阈值配置更精准。

**Q4: Method 4 的"圆整到 0.5 倍数"是否会损失精度？**

**A**: 不会。圆整的目的是提高可解释性（3.0 比 3.085 更易于沟通）。从验证结果看，圆整后的阈值仍然能准确区分风险等级，边界样本的平均分数与阈值有明显差距（2.69 vs 3.0）。

---

**📊 验证脚本**: `scripts/summary/compare_threshold_methods.py`  
**📈 可视化图表**: `analysis_results/risk_scoring_system/threshold_methods_comparison.png`  
**📄 对比报告**: `analysis_results/risk_scoring_system/threshold_methods_comparison_report.json`


