# agents/state.py
# Pydantic data models + LangGraph TypedDict pipeline state.
#
# Design:
#   - PipelineState   : TypedDict consumed by LangGraph (supports Annotated reducers)
#   - Everything else : Pydantic BaseModel used for per-node I/O validation
#
# DataFrames are stored directly in state (arbitrary_types_allowed is not needed
# for TypedDict; Pydantic models that hold DataFrames use model_config below).

from __future__ import annotations

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


# ── Shared data structures ────────────────────────────────────────────────────

class PatchNote(BaseModel):
    """A single Steam news / patch-note item."""
    gid: str
    title: str
    url: str
    published_at: datetime
    contents: str   # truncated to 2 000 chars in scraper


class SentimentStats(BaseModel):
    """Aggregate sentiment metrics for one review window."""
    mean_compound: float          # VADER compound mean  [-1, 1]
    positive_pct:  float          # fraction of reviews  ≥ +0.05
    neutral_pct:   float          # fraction in (-0.05, +0.05)
    negative_pct:  float          # fraction ≤ -0.05
    n_reviews:     int


class TopicCount(BaseModel):
    label: str
    count: int
    pct:   float   # fraction of post-update reviews assigned this label


class RiskSignal(BaseModel):
    rule:        str
    severity:    Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    value:       float     # observed metric value
    threshold:   float     # threshold that was crossed
    description: str


class RecommendationItem(BaseModel):
    priority:        Literal["P1", "P2", "P3"]
    issue:           str
    action:          str
    expected_impact: str


class Recommendations(BaseModel):
    """Final output of the recommendation agent."""
    narrative: str                         # 2-3 paragraph executive summary
    items:     list[RecommendationItem]    # structured priority list


# ── Per-node I/O schemas (validated at each node boundary) ───────────────────

class ScraperOutput(BaseModel):
    appid:            str
    game_name:        str
    n_pre_reviews:    int
    n_post_reviews:   int
    n_patch_notes:    int
    n_event_comments: int = 0   # comments fetched from the update announcement


class CleaningOutput(BaseModel):
    n_cleaned:          int
    n_flagged:          int


class EventAnalysisResult(BaseModel):
    """LLM-derived analysis of event comments (Map-Reduce output).

    Produced directly by core/event_analysis.analyse_event_comments().
    Consumed by: core/analysis (risk signals), recommendation_agent, UI pages.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_comments:        int
    positive_count:    int = 0
    negative_count:    int = 0
    neutral_count:     int = 0
    positive_pct:      float
    negative_pct:      float
    neutral_pct:       float
    top_themes:        list[dict]     # [{"label": ..., "count": ...}]
    representative_quotes: list[dict] # [{"text": ..., "sentiment": ...}]
    llm_summary:       str            # qualitative assessment from Reduce step
    patch_context:     str = ""       # patch notes used as context (truncated)


class ReviewAnalysisResult(BaseModel):
    """VADER-based analysis of reviews + disagreement detection."""
    sentiment:         SentimentStats
    n_disagreements:   int = 0        # VADER-positive but voted_up=False
    disagreement_pct:  float = 0.0
    distribution_summary: str = ""    # raw data distribution for LLM context


class AnalysisOutput(BaseModel):
    pre_sentiment:    SentimentStats            # baseline (pre-update reviews)
    review_analysis:  Optional[ReviewAnalysisResult] = None  # post-update reviews (VADER path)
    event_analysis:   Optional[EventAnalysisResult]  = None  # event comments (LLM path)
    sentiment_delta:  float              # post negative_pct - pre negative_pct (positive = worsening)
    top_topics:       list[TopicCount]
    risk_signals:     list[RiskSignal]
    overall_risk:     Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    gray_zone_count:  int                # reviews needing LLM sentiment re-score


# ── LangGraph pipeline state ──────────────────────────────────────────────────
# TypedDict keeps full compatibility with LangGraph's state management.
# List fields use Annotated[list, add] so each node *appends* rather than
# overwrites (standard LangGraph reducer pattern).

class PipelineState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    appid:       str
    update_date: datetime
    pre_days:    int       # days before update → baseline window
    post_days:   int       # days after  update → impact  window

    # ── Scraped ────────────────────────────────────────────────────────────
    game_name:      str
    pre_reviews:    Optional[pd.DataFrame]   # columns: review_id, review_content,
    post_reviews:   Optional[pd.DataFrame]   #   voted_up, timestamp, playtime_hours
    patch_notes:    list[PatchNote]
    event_comments: Optional[pd.DataFrame]   # comments under the specific update
                                             #   announcement; same column schema
                                             #   plus a 'source' column = "event_comment"

    # ── Cleaned (post-update reviews only) ─────────────────────────────────
    cleaned_reviews: Optional[pd.DataFrame]

    # ── Analysis ───────────────────────────────────────────────────────────
    analysis: Optional[AnalysisOutput]

    # ── LLM outputs ───────────────────────────────────────────────────────
    recommendations: Optional[Recommendations]

    # ── Pipeline metadata ───────────────────────────────────────────────────
    current_step: str
    errors:       Annotated[list[str], add]
