# Steam MAS 系统重构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有 Steam 评论分析 pipeline 重构为 LangGraph 驱动的多智能体系统，加入 LLM 复查机制，并提供 Streamlit 操作界面。

**Architecture:** 固定 pipeline 处理确定性工作（清洗/特征提取/推理），LLM Agent 在三个关键节点介入审查不确定数据；LangGraph 作为编排层，不重写已有业务逻辑。

**Tech Stack:** LangGraph, Claude API (claude-sonnet-4-6), Streamlit, Pandas, 现有 ML 模型（DistilBERT sentiment/topic, risk scoring）

---

## Phase 1: 项目基础结构

### Task 1: 创建目录结构

**Files:**
- Create: `agents/`
- Create: `core/`
- Create: `ui/`
- Create: `tests/`

**Step 1: 创建目录**

```powershell
New-Item -ItemType Directory agents, core, ui, tests
New-Item -ItemType File agents/__init__.py, core/__init__.py, ui/__init__.py, tests/__init__.py
```

**Step 2: 验证**

```powershell
Get-ChildItem -Directory | Select-Object Name
```

Expected: 看到 agents, core, ui, tests

---

### Task 2: 创建 config.py（替换所有硬编码路径）

**Files:**
- Create: `config.py`

**Step 1: 写 config.py**

```python
from pathlib import Path

BASE_DIR = Path(__file__).parent

# 数据路径
DATA_LABEL_DIR = BASE_DIR / "data_label"
DATA_NOLABEL_DIR = BASE_DIR / "data_nolabel"
FEATURES_DIR = BASE_DIR / "features"
ANALYSIS_DIR = BASE_DIR / "analysis_results"
MODELS_DIR = BASE_DIR / "models"

# 每个 genre 的组合数据路径
COMBINED_LABEL = {
    "fps":      DATA_LABEL_DIR / "combined" / "combined_fps_reviews.xlsx",
    "leisure":  DATA_LABEL_DIR / "combined" / "combined_leisure_reviews.xlsx",
    "strategy": DATA_LABEL_DIR / "combined" / "combined_strategy_reviews.xlsx",
}

NEWS_DIR = DATA_NOLABEL_DIR / "cleaned"

# LLM 设置
LLM_MODEL = "claude-sonnet-4-6"
LLM_REVIEW_THRESHOLD = 0.05   # flagged 行占比超过 5% 触发 LLM 复查
RISK_GRAY_ZONE = (1.5, 3.0)   # 风险分灰色地带触发 LLM

# 推理设置
INFERENCE_WINDOW_HOURS = 48
BASELINE_DAYS = 7

# 告警阈值
ALERT_THRESHOLDS = {
    "negative_sentiment_pct": 0.60,
    "critical_risk_pct": 0.05,
    "review_rate_multiplier": 3.0,
}
```

**Step 2: 验证**

```powershell
python -c "from config import BASE_DIR; print(BASE_DIR)"
```

Expected: 打印 BAP 目录路径，无报错

---

### Task 3: 创建 requirements.txt

**Files:**
- Create: `requirements.txt`

**Step 1: 写文件**

```
# Core ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
transformers>=4.35.0
torch>=2.0.0
sentence-transformers>=2.2.0

# NLP
nltk>=3.8.0
textstat>=0.7.0
vaderSentiment>=3.3.2

# Agent framework
langgraph>=0.2.0
langchain-anthropic>=0.2.0
anthropic>=0.40.0

# UI
streamlit>=1.30.0
plotly>=5.18.0

# Utils
joblib>=1.3.0
openpyxl>=3.1.0
pyarrow>=14.0.0
networkx>=3.2.0
tqdm>=4.66.0
```

**Step 2: 安装并验证**

```powershell
pip install -r requirements.txt
python -c "import langgraph; import anthropic; import streamlit; print('OK')"
```

---

## Phase 2: Core 业务逻辑层（从 scripts/ 提炼）

### Task 4: 创建 core/cleaning.py

**Files:**
- Create: `core/cleaning.py`
- Reference: `scripts/cleaning/review_data_cleaning_.py`

**Step 1: 写 cleaning.py 骨架**

```python
import pandas as pd
from pathlib import Path
from config import LLM_REVIEW_THRESHOLD

def run_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (cleaned_df, flagged_df)
    flagged_df 是需要 LLM 复查的疑似行
    """
    # 1. 去重
    df = df.drop_duplicates(subset=["review_text"])
    
    # 2. 长度过滤
    df["text_len"] = df["review_text"].str.len()
    flagged = df[df["text_len"] < 10].copy()
    df = df[df["text_len"] >= 10].copy()
    
    # 3. 语言过滤（非英文标记为 flagged）
    # TODO: 整合 scripts/cleaning 中的语言检测逻辑
    
    return df, flagged

def should_trigger_llm_review(flagged_df: pd.DataFrame, total: int) -> bool:
    if total == 0:
        return False
    return len(flagged_df) / total > LLM_REVIEW_THRESHOLD
```

**Step 2: 验证**

```python
# tests/test_cleaning.py
import pandas as pd
from core.cleaning import run_cleaning, should_trigger_llm_review

def test_cleaning_removes_short_text():
    df = pd.DataFrame({"review_text": ["ok", "this is a great game with many features"]})
    cleaned, flagged = run_cleaning(df)
    assert len(flagged) == 1
    assert flagged.iloc[0]["review_text"] == "ok"

def test_llm_trigger_threshold():
    flagged = pd.DataFrame({"review_text": ["x"] * 10})
    assert should_trigger_llm_review(flagged, 100) is True   # 10% > 5%
    assert should_trigger_llm_review(flagged, 1000) is False  # 1% < 5%
```

```powershell
python -m pytest tests/test_cleaning.py -v
```

---

### Task 5: 创建 core/inference.py

**Files:**
- Create: `core/inference.py`
- Reference: `scripts/inference/batch_inference.py`（修复占位符问题）

**Step 1: 写 inference.py**

```python
import pandas as pd
import pickle
from pathlib import Path
from config import MODELS_DIR

class InferenceEngine:
    def __init__(self, genre: str):
        self.genre = genre
        self.sentiment_model = None
        self.topic_model = None
        self._load_models()
    
    def _load_models(self):
        model_dir = MODELS_DIR / self.genre
        sentiment_path = model_dir / "sentiment_advanced_bert.pkl"
        topic_path = model_dir / "topic_advanced_bert.pkl"
        
        if sentiment_path.exists():
            with open(sentiment_path, "rb") as f:
                self.sentiment_model = pickle.load(f)
        
        if topic_path.exists():
            with open(topic_path, "rb") as f:
                self.topic_model = pickle.load(f)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        if self.sentiment_model:
            result["sentiment_pred"] = self.sentiment_model.predict(df["review_text"])
        else:
            result["sentiment_pred"] = "unknown"
        
        if self.topic_model:
            result["topic_pred"] = self.topic_model.predict(df["review_text"])
        else:
            result["topic_pred"] = "unknown"
        
        # Risk score 使用现有规则引擎（不依赖 pkl）
        result["risk_score"] = self._compute_risk(result)
        result["risk_level"] = result["risk_score"].apply(self._risk_label)
        
        return result
    
    def _compute_risk(self, df: pd.DataFrame) -> pd.Series:
        # 从 scripts/inference/batch_inference.py 移植规则引擎
        score = pd.Series(0.0, index=df.index)
        if "sentiment_pred" in df.columns:
            score += (df["sentiment_pred"] == "negative").astype(float) * 2.0
        return score.clip(0, 10)
    
    def _risk_label(self, score: float) -> str:
        if score >= 3.0: return "critical"
        if score >= 2.0: return "high"
        if score >= 1.0: return "medium"
        return "low"
```

**Step 2: 验证**

```powershell
python -c "from core.inference import InferenceEngine; e = InferenceEngine('fps'); print('loaded OK')"
```

---

## Phase 3: LangGraph Agent 层

### Task 6: 定义共享状态

**Files:**
- Create: `agents/state.py`

**Step 1: 写状态定义**

```python
from typing import TypedDict, Optional
import pandas as pd

class SteamAnalysisState(TypedDict):
    genre: str                          # fps / leisure / strategy
    raw_data: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]
    flagged_data: Optional[pd.DataFrame]
    llm_review_result: Optional[dict]   # LLM 对 flagged 行的判断
    features: Optional[pd.DataFrame]
    inference_result: Optional[pd.DataFrame]
    alerts: list
    llm_review_log: list                # 给 UI 展示的 LLM 决策记录
    current_step: str
    errors: list
```

**Step 2: 验证**

```powershell
python -c "from agents.state import SteamAnalysisState; print('state OK')"
```

---

### Task 7: 实现 CleaningAgent（含 LLM 触发逻辑）

**Files:**
- Create: `agents/cleaning_agent.py`

**Step 1: 写 CleaningAgent**

```python
from agents.state import SteamAnalysisState
from core.cleaning import run_cleaning, should_trigger_llm_review

def cleaning_node(state: SteamAnalysisState) -> SteamAnalysisState:
    df = state["raw_data"]
    cleaned, flagged = run_cleaning(df)
    
    return {
        **state,
        "cleaned_data": cleaned,
        "flagged_data": flagged,
        "current_step": "cleaning_done",
    }

def should_run_llm_review(state: SteamAnalysisState) -> str:
    """LangGraph 条件边：决定下一步走 LLM 复查还是直接进特征工程"""
    flagged = state.get("flagged_data")
    total = len(state.get("cleaned_data", [])) + len(flagged or [])
    
    if flagged is not None and should_trigger_llm_review(flagged, total):
        return "llm_review"
    return "feature_engineering"
```

---

### Task 8: 实现 LLM Review Agent（Claude API 核心）

**Files:**
- Create: `agents/llm_review_agent.py`

**Step 1: 写 LLM Review Agent**

```python
import anthropic
from agents.state import SteamAnalysisState
from config import LLM_MODEL

client = anthropic.Anthropic()

REVIEW_SYSTEM_PROMPT = """你是一个 Steam 游戏评论数据质量审核专家。
你会收到被自动清洗程序标记为"可疑"的评论列表，需要判断每条是否应该保留。

对每条评论，回答：
- keep: 保留（内容有效，只是语言简短/特殊）
- remove: 删除（垃圾内容/重复/无意义）
- uncertain: 不确定（需要人工复核）

以 JSON 格式返回：{"results": [{"index": 0, "decision": "keep", "reason": "..."}]}"""

def llm_review_node(state: SteamAnalysisState) -> SteamAnalysisState:
    flagged = state["flagged_data"]
    if flagged is None or len(flagged) == 0:
        return {**state, "current_step": "llm_review_done"}
    
    # 批量发送（最多 50 条）
    samples = flagged.head(50)
    review_input = "\n".join(
        f"[{i}] {row['review_text'][:200]}"
        for i, (_, row) in enumerate(samples.iterrows())
    )
    
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=2048,
        system=REVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"请审核以下评论：\n{review_input}"}],
    )
    
    import json
    result = json.loads(response.content[0].text)
    
    # 根据 LLM 决定合并数据
    keep_indices = {r["index"] for r in result["results"] if r["decision"] == "keep"}
    rows_to_keep = samples.iloc[list(keep_indices)]
    
    merged_cleaned = pd.concat([state["cleaned_data"], rows_to_keep], ignore_index=True)
    
    log_entry = {
        "step": "cleaning_review",
        "flagged_count": len(flagged),
        "kept": len(keep_indices),
        "removed": len(flagged) - len(keep_indices),
        "details": result["results"],
    }
    
    return {
        **state,
        "cleaned_data": merged_cleaned,
        "llm_review_result": result,
        "llm_review_log": state.get("llm_review_log", []) + [log_entry],
        "current_step": "llm_review_done",
    }
```

**Step 2: 验证（需要 ANTHROPIC_API_KEY）**

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python -c "from agents.llm_review_agent import llm_review_node; print('LLM agent OK')"
```

---

### Task 9: 实现 InferenceAgent 和 MonitorAgent

**Files:**
- Create: `agents/inference_agent.py`
- Create: `agents/monitor_agent.py`

**Step 1: inference_agent.py**

```python
from agents.state import SteamAnalysisState
from core.inference import InferenceEngine
from config import LLM_MODEL, RISK_GRAY_ZONE
import anthropic, json

client = anthropic.Anthropic()

def inference_node(state: SteamAnalysisState) -> SteamAnalysisState:
    engine = InferenceEngine(state["genre"])
    result = engine.predict(state["features"])
    
    return {**state, "inference_result": result, "current_step": "inference_done"}

def should_run_risk_review(state: SteamAnalysisState) -> str:
    """灰色地带风险分触发 LLM 复核"""
    result = state.get("inference_result")
    if result is None:
        return "monitor"
    low, high = RISK_GRAY_ZONE
    gray_zone = result[(result["risk_score"] >= low) & (result["risk_score"] < high)]
    return "risk_llm_review" if len(gray_zone) > 0 else "monitor"
```

**Step 2: monitor_agent.py**

```python
from agents.state import SteamAnalysisState
from config import ALERT_THRESHOLDS, LLM_MODEL
import anthropic, json

client = anthropic.Anthropic()

def monitor_node(state: SteamAnalysisState) -> SteamAnalysisState:
    result = state.get("inference_result")
    alerts = []
    
    if result is not None:
        neg_pct = (result["sentiment_pred"] == "negative").mean()
        critical_pct = (result["risk_level"] == "critical").mean()
        
        if neg_pct > ALERT_THRESHOLDS["negative_sentiment_pct"]:
            alerts.append({"rule": "R1", "severity": "HIGH",
                          "message": f"负面评论占比 {neg_pct:.1%}，超过阈值"})
        
        if critical_pct > ALERT_THRESHOLDS["critical_risk_pct"]:
            alerts.append({"rule": "R2", "severity": "CRITICAL",
                          "message": f"Critical 风险占比 {critical_pct:.1%}"})
    
    # 有 CRITICAL 告警则让 LLM 生成摘要
    if any(a["severity"] == "CRITICAL" for a in alerts):
        summary = _generate_crisis_summary(alerts)
        state["llm_review_log"] = state.get("llm_review_log", []) + [{
            "step": "crisis_summary", "summary": summary
        }]
    
    return {**state, "alerts": alerts, "current_step": "done"}

def _generate_crisis_summary(alerts: list) -> str:
    client_inst = anthropic.Anthropic()
    response = client_inst.messages.create(
        model=LLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content":
            f"以下是游戏评论监控系统检测到的告警，请用2-3句话生成中文危机摘要：\n{json.dumps(alerts, ensure_ascii=False)}"}]
    )
    return response.content[0].text
```

---

### Task 10: 组装 LangGraph 主图

**Files:**
- Create: `graph.py`

**Step 1: 写主图**

```python
from langgraph.graph import StateGraph, END
from agents.state import SteamAnalysisState
from agents.cleaning_agent import cleaning_node, should_run_llm_review
from agents.llm_review_agent import llm_review_node
from agents.inference_agent import inference_node, should_run_risk_review
from agents.monitor_agent import monitor_node

def build_graph():
    g = StateGraph(SteamAnalysisState)
    
    # 注册节点
    g.add_node("cleaning",          cleaning_node)
    g.add_node("llm_review",        llm_review_node)
    g.add_node("feature_engineering", lambda s: {**s, "features": s["cleaned_data"], "current_step": "features_done"})
    g.add_node("inference",         inference_node)
    g.add_node("risk_llm_review",   llm_review_node)   # 复用同一个 LLM review 节点
    g.add_node("monitor",           monitor_node)
    
    # 入口
    g.set_entry_point("cleaning")
    
    # 条件边：清洗后 → LLM复查 或 特征工程
    g.add_conditional_edges("cleaning", should_run_llm_review, {
        "llm_review":         "llm_review",
        "feature_engineering": "feature_engineering",
    })
    g.add_edge("llm_review",         "feature_engineering")
    g.add_edge("feature_engineering", "inference")
    
    # 条件边：推理后 → 风险LLM复查 或 监控
    g.add_conditional_edges("inference", should_run_risk_review, {
        "risk_llm_review": "risk_llm_review",
        "monitor":         "monitor",
    })
    g.add_edge("risk_llm_review", "monitor")
    g.add_edge("monitor", END)
    
    return g.compile()

app = build_graph()
```

**Step 2: 验证图结构**

```python
from graph import app
print(app.get_graph().draw_ascii())
```

---

## Phase 4: Streamlit UI

### Task 11: 主 UI 框架

**Files:**
- Create: `ui/app.py`

**Step 1: 写 app.py**

```python
import streamlit as st
import pandas as pd
from graph import app
from agents.state import SteamAnalysisState
from config import COMBINED_LABEL

st.set_page_config(page_title="Steam MAS 分析系统", layout="wide")

pages = {
    "Pipeline 控制台": "ui/pages/pipeline.py",
    "数据监控":        "ui/pages/monitor.py",
    "LLM 审查日志":   "ui/pages/llm_log.py",
    "危机告警":        "ui/pages/alerts.py",
}

page = st.sidebar.selectbox("页面", list(pages.keys()))
```

**Step 2: 创建 Pipeline 控制台页面**

```python
# ui/pages/pipeline.py
import streamlit as st
import pandas as pd
from graph import app
from config import COMBINED_LABEL

def render():
    st.title("Pipeline 控制台")
    
    genre = st.selectbox("选择游戏类型", ["fps", "leisure", "strategy"])
    
    if st.button("运行分析", type="primary"):
        data_path = COMBINED_LABEL[genre]
        df = pd.read_excel(data_path)
        
        initial_state = {
            "genre": genre,
            "raw_data": df,
            "cleaned_data": None,
            "flagged_data": None,
            "llm_review_result": None,
            "features": None,
            "inference_result": None,
            "alerts": [],
            "llm_review_log": [],
            "current_step": "start",
            "errors": [],
        }
        
        with st.spinner("运行中..."):
            for step_output in app.stream(initial_state):
                node_name = list(step_output.keys())[0]
                st.write(f"✅ {node_name} 完成")
        
        st.success("分析完成！")
        st.session_state["last_result"] = step_output
```

**Step 3: 创建 LLM 审查日志页面（简历亮点）**

```python
# ui/pages/llm_log.py
import streamlit as st

def render():
    st.title("LLM 审查日志")
    st.caption("展示 LLM 在哪些节点介入了决策，以及给出的判断")
    
    result = st.session_state.get("last_result", {})
    state = list(result.values())[0] if result else {}
    log = state.get("llm_review_log", [])
    
    if not log:
        st.info("暂无 LLM 审查记录，请先运行 Pipeline")
        return
    
    for entry in log:
        with st.expander(f"[{entry['step']}] 审查记录"):
            if entry["step"] == "cleaning_review":
                st.metric("标记数量", entry["flagged_count"])
                st.metric("保留", entry["kept"])
                st.metric("删除", entry["removed"])
                st.json(entry["details"])
            elif entry["step"] == "crisis_summary":
                st.warning(entry["summary"])
```

---

## Phase 5: 集成测试

### Task 12: 端到端冒烟测试

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: 写端到端测试**

```python
import pandas as pd
from graph import app

def test_pipeline_runs_without_crash():
    """用小样本验证整个图能跑通"""
    mock_df = pd.DataFrame({
        "review_text": [
            "Great game!",
            "ok",  # 会被 flagged
            "Amazing graphics and smooth gameplay experience.",
            "terrible",
        ] * 5,
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
        "author_steamid": range(20),
    })
    
    initial_state = {
        "genre": "fps",
        "raw_data": mock_df,
        "cleaned_data": None,
        "flagged_data": None,
        "llm_review_result": None,
        "features": None,
        "inference_result": None,
        "alerts": [],
        "llm_review_log": [],
        "current_step": "start",
        "errors": [],
    }
    
    result = app.invoke(initial_state)
    assert result["current_step"] == "done"
    assert result["cleaned_data"] is not None
```

**Step 2: 运行测试**

```powershell
python -m pytest tests/test_e2e.py -v
```

---

## 执行顺序建议

```
Phase 1 (Task 1-3)  →  基础结构，30 分钟内完成
Phase 2 (Task 4-5)  →  Core 层，优先 cleaning，inference 可以先用简版
Phase 3 (Task 6-10) →  Agent + 图，这是核心，需要 ANTHROPIC_API_KEY
Phase 4 (Task 11)   →  UI，最后做，做完就能 demo
Phase 5 (Task 12)   →  贯穿整个过程，每个 Phase 完成后跑一次
```

## 注意事项

- `ANTHROPIC_API_KEY` 需要设置为环境变量才能运行 LLM 节点
- 项目目前不是 git repo，建议先 `git init` 再开始
- `core/inference.py` 里的 BERT 模型加载依赖已训练好的 `.pkl` 文件，如果文件不存在会自动降级为规则引擎
- Streamlit 运行：`streamlit run ui/app.py`
