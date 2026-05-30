# Steam 更新影响分析系统

> 基于 LangGraph + Claude API 构建的多智能体系统，自动分析 Steam 游戏更新对玩家情感的影响，生成可操作的风险报告与优化建议。

---

## 项目概述

游戏每次更新后，开发团队面临一个核心问题：**这次更新到底让玩家更满意了，还是更差了？**

本系统通过以下方式回答这个问题：

1. 抓取更新前后的 Steam 评论，进行情感对比
2. 从更新公告页直接抓取玩家评论（高信号数据源）
3. 识别情感下滑、负评激增、话题异常等风险信号
4. 由 Claude Sonnet 生成结构化执行报告与优先级行动项

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **多源数据采集** | Steam 评论 API + 补丁日志 + 公告页玩家评论 |
| **本地 SQLite 缓存** | 支持历史数据查询，突破 Steam API 30 天限制 |
| **智能清洗** | 规则过滤 + Claude Haiku 二次审核可疑评论 |
| **情感分析** | VADER 快速评分，灰区评论由 LLM 重新评分 |
| **话题分类** | Claude Haiku 批量分类（性能/玩法/内容/Bug/付费等） |
| **风险评估** | 规则引擎检测情感骤降、负评激增、评论量异常 |
| **执行报告** | Claude Sonnet 生成 P1/P2/P3 优先级行动清单 |
| **可视化看板** | Streamlit 交互界面，含情感趋势图、话题分布、风险仪表盘 |

---

## 系统架构

```
用户输入（游戏 + 更新日期）
         │
    ┌────▼────┐
    │ Scraper │  ← Steam 评论 API + 公告页评论 + 本地缓存
    └────┬────┘
         │
    ┌────▼────┐
    │Cleaning │  ← 规则清洗 → 可选 LLM 二次审核
    └────┬────┘
         │
    ┌────▼────┐
    │Analysis │  ← VADER 情感 + Claude 话题分类 + 风险评分
    └────┬────┘
         │ 灰区评论
    ┌────▼──────────┐
    │LLM Sentiment  │  ← Claude Haiku 重评分
    └────┬──────────┘
         │
    ┌────▼──────────────┐
    │  Recommendation   │  ← Claude Sonnet 生成报告
    └────┬──────────────┘
         │
      Streamlit UI
```

**技术栈**：LangGraph · Pydantic v2 · VADER · Claude API · SQLite · Streamlit · Plotly

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 项目根目录创建 .env 文件
ANTHROPIC_API_KEY=your_key_here
```

### 3. 启动 UI

```bash
streamlit run ui/app.py
```

### 4. （可选）初始化历史数据缓存

在侧边栏"添加游戏"时，可选择三种初始化方式：

- **抓取最新 N 条评论** — 适合快速开始（推荐 2–5 万条）
- **全量抓取** — 无上限，适合深度历史分析（耗时较长）
- **上传本地数据文件** — 支持 CSV / Excel，自动识别旧版 schema

---

## 项目结构

```
BAP/
├── agents/               # LangGraph 节点
│   ├── state.py          # Pydantic 状态模型
│   ├── scraper_agent.py  # 数据采集节点
│   ├── cleaning_agent.py # 清洗节点
│   ├── analysis_agent.py # 分析节点
│   ├── llm_review_agent.py
│   ├── llm_sentiment_agent.py
│   └── recommendation_agent.py
├── core/                 # 纯函数业务逻辑
│   ├── scraper.py        # Steam API 客户端
│   ├── local_cache.py    # SQLite 缓存层
│   ├── bulk_scraper.py   # 批量抓取工具
│   ├── event_comments.py # 公告页评论抓取
│   ├── features.py       # NLP 特征提取（VADER）
│   ├── cleaning.py       # 评论清洗规则
│   └── analysis.py       # 情感统计 + 话题分类 + 风险评分
├── ui/
│   ├── app.py            # Streamlit 主入口
│   └── pages/
│       ├── dashboard.py  # 可视化看板
│       └── report.py     # 分析报告页
├── graph.py              # LangGraph 图构建
├── config.py             # 全局配置
├── import_cache.py       # 历史数据导入脚本
├── tests/
│   └── test_e2e.py       # E2E + 单元测试
└── requirements.txt
```

---

## 设计原则

- **无 GPU 依赖**：NLP 全部使用 VADER（规则）+ Claude API（按需），无需本地模型
- **通用性**：不针对特定游戏类型，支持任意 Steam 游戏
- **数据不泄露**：预更新/后更新窗口边界精确控制（1 秒间隔），避免时序污染
- **成本控制**：Claude Haiku 处理高频低难度任务，Sonnet 只用于最终报告；所有系统提示启用 Prompt Caching
- **状态校验**：Pydantic v2 在每个 Agent 边界做 I/O 校验

---

## 内置游戏目录

| 游戏 | AppID | 开发商 |
|------|-------|--------|
| Counter-Strike 2 | 730 | Valve |
| Dota 2 | 570 | Valve |
| Elden Ring | 1245620 | FromSoftware |
| Cyberpunk 2077 | 1091500 | CD Projekt Red |
| Baldur's Gate 3 | 1086940 | Larian Studios |
| Stardew Valley | 413150 | ConcernedApe |
| Hollow Knight | 367520 | Team Cherry |
| God of War | 1593500 | Santa Monica Studio |

---

## 测试

```bash
python -m pytest tests/test_e2e.py -v
```

覆盖：E2E 管线冒烟测试（全 Mock）+ 特征提取 + 清洗规则 + 情感统计边界

