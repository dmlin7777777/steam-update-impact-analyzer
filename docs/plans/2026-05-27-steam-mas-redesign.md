# Steam Update Impact Analyzer — Implementation Plan

**Goal:** A general-purpose MAS that, given a game AppID and update date, fetches Steam reviews
before/after the update, runs NLP analysis, and uses Claude to generate a root-cause + recommendation report.

**Key design decisions:**
- No pre-trained genre-specific models — everything runs on any Steam game
- VADER for fast sentiment; Claude (haiku) for ambiguous/topic tagging; Claude (sonnet) for recommendations
- Pydantic I/O contracts at every agent boundary
- LangGraph for orchestration; Streamlit for UI

---

## Architecture

```
[scraper_node]
      ↓
[cleaning_node] ──(flagged > 5%)──→ [llm_review_node]
      ↓                                     ↓
[analysis_node] ←────────────────────────┘
      ↓
      ├──(gray-zone reviews exist)──→ [llm_sentiment_node]
      ↓                                     ↓
[risk_node] ←────────────────────────────┘
      ↓
[recommendation_node]  (sonnet: patch notes + analysis → report)
      ↓
    [END]
```

---

## Task List

### Task 1 — agents/state.py
Pydantic data models + TypedDict pipeline state with LangGraph reducers.

### Task 2 — core/scraper.py
Steam Reviews API + News API client. Cursor pagination, date filtering, retry logic.

### Task 3 — core/features.py
VADER sentiment scoring + textstat readability + basic text stats. No GPU.

### Task 4 — core/analysis.py
Sentiment trend (pre vs post), topic batching via Claude haiku, Z-score anomaly,
risk rule engine → AnalysisOutput Pydantic model.

### Task 5 — agents/scraper_agent.py
Wraps core/scraper. Validates ScraperOutput. Sets game_name, pre/post reviews, patch_notes.

### Task 6 — agents/cleaning_agent.py
Calls core/cleaning.run_cleaning. Validates CleaningOutput. Conditional edge → llm_review.

### Task 7 — agents/llm_review_agent.py
Claude haiku reviews flagged rows. Validates LLMReviewOutput. Merges kept rows back.

### Task 8 — agents/analysis_agent.py
Calls core/analysis. Validates AnalysisOutput. Conditional edge → llm_sentiment if gray-zone.

### Task 9 — agents/llm_sentiment_agent.py
Claude haiku re-scores ambiguous reviews. Updates sentiment stats in state.

### Task 10 — agents/recommendation_agent.py
Claude sonnet: patch notes + AnalysisOutput → Recommendations (narrative + structured items).

### Task 11 — graph.py
Assemble all nodes + conditional edges. Export compiled `app`.

### Task 12 — ui/app.py + ui/pages/
Streamlit: game list sidebar, update announcements, dashboard charts, report view.

### Task 13 — tests/test_e2e.py
Smoke test with mock Steam responses (no live API call).

---

## Column conventions
- Raw review field from Steam API: `review` → normalised to `review_content` in scraper
- Cleaned/processed text: `review_content_processed` (added by features.py)
- Timestamp: `timestamp` (UTC-aware datetime)
