# core/analysis.py
# Post-update impact analysis: sentiment stats, topic tagging, risk scoring.
#
# Two analysis paths feed into this module:
#
#   1. Reviews (post-update):  VADER sentiment + voted_up disagreement detection.
#      VADER is fast but unreliable on sarcasm/slang. Disagreement detection
#      (VADER-positive but voted_up=False) surfaces likely misclassifications.
#      A distribution summary of raw data is generated for downstream LLM context.
#
#   2. Event comments:  analysed by core/event_analysis.py (Map-Reduce LLM path).
#      This module does NOT process event comments — it receives the finished
#      EventCommentAnalysis result and incorporates it into risk scoring.
#
# Topic tagging still uses Claude Haiku for reviews (event comment topics come
# from the Map-Reduce pipeline).

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from agents.state import (
    AnalysisOutput,
    EventAnalysisResult,
    ReviewAnalysisResult,
    RiskSignal,
    SentimentStats,
    TopicCount,
)
from config import (
    ALERT_THRESHOLDS,
    LLM_MODEL_LIGHT,
    TOPIC_LABELS,
    VADER_GRAY_LO,
)
from core.llm import chat as llm_chat


# ── Sentiment aggregation (reviews only — event comments use LLM path) ───────

def compute_sentiment_stats(df: pd.DataFrame) -> SentimentStats:
    """
    Aggregate VADER scores into SentimentStats.
    Requires columns: vader_compound, sentiment_label.
    """
    if df.empty:
        return SentimentStats(
            mean_compound=0.0, positive_pct=0.0,
            neutral_pct=0.0,   negative_pct=0.0, n_reviews=0,
        )

    label_counts = df["sentiment_label"].value_counts(normalize=True)
    return SentimentStats(
        mean_compound=float(df["vader_compound"].mean()),
        positive_pct= float(label_counts.get("positive", 0.0)),
        neutral_pct=  float(label_counts.get("neutral",  0.0)),
        negative_pct= float(label_counts.get("negative", 0.0)),
        n_reviews=    len(df),
    )


# ── Disagreement detection (VADER vs voted_up) ──────────────────────────────

def detect_disagreements(df: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    """
    Find reviews where VADER sentiment contradicts the player's own vote.

    VADER compound > 0.05 but voted_up=False  → likely sarcasm/irony.
    VADER compound < -0.05 but voted_up=True  → likely understated praise.

    Returns:
        (n_disagreements, disagreement_pct, disagreement_rows_df)
    """
    if df.empty or "vader_compound" not in df.columns or "voted_up" not in df.columns:
        return 0, 0.0, pd.DataFrame()

    disagree = (
        ((df["vader_compound"] > 0.05) & (~df["voted_up"])) |
        ((df["vader_compound"] < -0.05) & (df["voted_up"]))
    )
    n = int(disagree.sum())
    pct = round(n / max(len(df), 1), 4)
    return n, pct, df[disagree]


# ── Distribution summary (raw data description for LLM context) ─────────────

def build_distribution_summary(df: pd.DataFrame) -> str:
    """
    Generate a factual description of the raw review data distribution.
    No interpretation — just numbers the LLM can reason about.
    """
    if df.empty:
        return "No reviews in this window."

    total = len(df)
    lines = [f"{total} reviews total."]

    # voted_up distribution
    if "voted_up" in df.columns:
        pos = int(df["voted_up"].sum())
        neg = total - pos
        lines.append(f"Recommended: {pos} ({pos/total:.0%}). Not recommended: {neg} ({neg/total:.0%}).")

    # Short text distribution
    if "review_content" in df.columns:
        lens = df["review_content"].str.len()
        short = df[lens < 10]
        if len(short) > 0:
            freq = short["review_content"].str.strip().value_counts().head(8)
            short_str = ", ".join(f'"{t}" ({c}x)' for t, c in freq.items())
            lines.append(f"{len(short)} reviews under 10 chars. Most common: {short_str}")

    # VADER distribution
    if "vader_compound" in df.columns:
        lines.append(
            f"VADER compound: mean={df['vader_compound'].mean():+.3f}, "
            f"median={df['vader_compound'].median():+.3f}, "
            f"std={df['vader_compound'].std():.3f}"
        )

    return "\n".join(lines)


# ── Topic tagging (Claude haiku, batched) — reviews only ─────────────────────

_TOPIC_SYSTEM = (
    "You are a game review analyst. Classify each review into exactly ONE topic "
    "from the list below. Reply with a JSON array — one label string per review, "
    "in the same order.\n\nTopics: " + ", ".join(TOPIC_LABELS)
)


def tag_topics(df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
    """
    Add a 'topic' column to df using Claude haiku batch classification.
    Falls back to 'other' on any error.
    Requires column: review_content_processed
    """
    if df.empty:
        df = df.copy()
        df["topic"] = pd.Series(dtype=str)
        return df

    texts   = df["review_content_processed"].tolist()
    labels: list[str] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        numbered = "\n".join(f"[{j}] {t[:300]}" for j, t in enumerate(chunk))

        try:
            raw = llm_chat(
                model=LLM_MODEL_LIGHT,
                system=_TOPIC_SYSTEM,
                user=numbered,
                max_tokens=512,
            )
            parsed: list[str] = json.loads(raw)
            parsed = [lbl if lbl in TOPIC_LABELS else "other" for lbl in parsed]
            if len(parsed) < len(chunk):
                parsed += ["other"] * (len(chunk) - len(parsed))
            labels.extend(parsed[: len(chunk)])
        except Exception:
            labels.extend(["other"] * len(chunk))

    df = df.copy()
    df["topic"] = labels
    return df


def aggregate_topics(df: pd.DataFrame) -> list[TopicCount]:
    """Count and rank topics from a tagged DataFrame."""
    if df.empty or "topic" not in df.columns:
        return []
    counts = df["topic"].value_counts()
    total  = len(df)
    return [
        TopicCount(label=label, count=int(cnt), pct=round(cnt / total, 4))
        for label, cnt in counts.items()
    ]


# ── Risk scoring ─────────────────────────────────────────────────────────────

def compute_risk_signals(
    pre: SentimentStats,
    review_analysis: Optional[ReviewAnalysisResult],
    event_analysis:  Optional[EventAnalysisResult],
) -> list[RiskSignal]:
    """
    Apply risk checks using BOTH data sources.

    Review-based signals (R1-R3): use VADER stats from review_analysis.
    Event-comment signals (R4-R5): use LLM-derived stats from event_analysis.
    """
    signals: list[RiskSignal] = []

    # ── Review-based signals ─────────────────────────────────────────────────
    if review_analysis is not None:
        post = review_analysis.sentiment
        delta = round(post.mean_compound - pre.mean_compound, 4)

        # R1 — Sentiment drop (reviews)
        thresh_drop = ALERT_THRESHOLDS["sentiment_drop"]
        if delta < -thresh_drop:
            signals.append(RiskSignal(
                rule="R1",
                severity=_drop_severity(delta),
                value=round(delta, 4),
                threshold=-thresh_drop,
                description=(
                    f"Review sentiment dropped {abs(delta):.3f} points "
                    f"(threshold {thresh_drop})"
                ),
            ))

        # R2 — Negative surge (reviews)
        thresh_neg = ALERT_THRESHOLDS["negative_surge_pct"]
        if post.negative_pct > thresh_neg:
            signals.append(RiskSignal(
                rule="R2",
                severity="HIGH" if post.negative_pct < 0.75 else "CRITICAL",
                value=round(post.negative_pct, 4),
                threshold=thresh_neg,
                description=(
                    f"{post.negative_pct:.1%} of post-update reviews are negative "
                    f"(threshold {thresh_neg:.0%})"
                ),
            ))

        # R3 — Review-volume spike
        if pre.n_reviews > 0:
            rate = post.n_reviews / pre.n_reviews
            thresh_rate = ALERT_THRESHOLDS["review_rate_multiplier"]
            if rate > thresh_rate:
                signals.append(RiskSignal(
                    rule="R3",
                    severity="MEDIUM",
                    value=round(rate, 2),
                    threshold=thresh_rate,
                    description=(
                        f"Review volume spiked {rate:.1f}x baseline "
                        f"(threshold {thresh_rate:.0f}x)"
                    ),
                ))

    # ── Event-comment signals ────────────────────────────────────────────────
    if event_analysis is not None:
        # R4 — Announcement negativity (LLM-assessed)
        if event_analysis.negative_pct > 0.50:
            sev = "CRITICAL" if event_analysis.negative_pct > 0.70 else "HIGH"
            signals.append(RiskSignal(
                rule="R4",
                severity=sev,
                value=round(event_analysis.negative_pct, 4),
                threshold=0.50,
                description=(
                    f"{event_analysis.negative_pct:.1%} of announcement comments are "
                    f"negative (LLM-assessed, {event_analysis.n_comments} comments)"
                ),
            ))

        # R5 — Announcement-vs-review divergence
        if review_analysis is not None:
            review_neg = review_analysis.sentiment.negative_pct
            event_neg  = event_analysis.negative_pct
            divergence = abs(event_neg - review_neg)
            if divergence > 0.20:
                signals.append(RiskSignal(
                    rule="R5",
                    severity="MEDIUM",
                    value=round(divergence, 4),
                    threshold=0.20,
                    description=(
                        f"Announcement comments ({event_neg:.0%} neg) diverge from "
                        f"reviews ({review_neg:.0%} neg) by {divergence:.0%}"
                    ),
                ))

    return signals


def _drop_severity(delta: float) -> str:
    if delta < -0.40:
        return "CRITICAL"
    if delta < -0.25:
        return "HIGH"
    return "MEDIUM"


def _overall_risk(signals: list[RiskSignal]) -> str:
    if not signals:
        return "LOW"
    severities = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    top = max(signals, key=lambda s: severities.get(s.severity, 0))
    return top.severity


# ── Orchestrating entry point ────────────────────────────────────────────────

def run_analysis(
    pre_df:          pd.DataFrame,
    post_df:         pd.DataFrame,
    event_analysis:  Optional[EventAnalysisResult] = None,
) -> AnalysisOutput:
    """
    Full analysis pipeline for one update event.

    Two independent paths:
      1. Reviews (post_df):   VADER stats + disagreement detection
      2. Event comments:      Already analysed via Map-Reduce LLM, passed in as
                              EventAnalysisResult (from core/event_analysis.py)

    Risk signals combine both paths. Topic tagging is applied to reviews only
    (event comment themes come from the Map-Reduce pipeline).
    """
    pre_stats = compute_sentiment_stats(pre_df)

    # ── Review analysis path ─────────────────────────────────────────────────
    review_result: Optional[ReviewAnalysisResult] = None
    if not post_df.empty:
        review_stats = compute_sentiment_stats(post_df)
        n_disagree, disagree_pct, _ = detect_disagreements(post_df)
        dist_summary = build_distribution_summary(post_df)

        review_result = ReviewAnalysisResult(
            sentiment=review_stats,
            n_disagreements=n_disagree,
            disagreement_pct=disagree_pct,
            distribution_summary=dist_summary,
        )

    # ── Sentiment delta (reviews only — event comments have no compound) ─────
    delta = 0.0
    if review_result is not None:
        delta = round(review_result.sentiment.mean_compound - pre_stats.mean_compound, 4)

    # ── Topic tagging on reviews ─────────────────────────────────────────────
    post_tagged = tag_topics(post_df) if not post_df.empty else post_df
    review_topics = aggregate_topics(post_tagged)

    # Merge event themes into topic list (event themes come from Map-Reduce)
    if event_analysis is not None:
        for theme in event_analysis.top_themes:
            label = theme.get("label", "")
            count = theme.get("count", 0)
            if label:
                review_topics.append(
                    TopicCount(label=f"[announcement] {label}", count=count, pct=0.0)
                )
        # Re-sort by count
        review_topics.sort(key=lambda t: t.count, reverse=True)

    # ── Risk signals (both paths) ────────────────────────────────────────────
    signals      = compute_risk_signals(pre_stats, review_result, event_analysis)
    overall_risk = _overall_risk(signals)

    # ── Gray-zone count (reviews only) ───────────────────────────────────────
    gray_zone_count = int(
        post_df["vader_compound"]
        .between(-VADER_GRAY_LO, VADER_GRAY_LO)
        .sum()
    ) if not post_df.empty and "vader_compound" in post_df.columns else 0

    # ── Convert EventCommentAnalysis dataclass → EventAnalysisResult pydantic
    event_result: Optional[EventAnalysisResult] = None
    if event_analysis is not None:
        event_result = EventAnalysisResult(
            n_comments=event_analysis.n_comments,
            positive_pct=event_analysis.positive_pct,
            negative_pct=event_analysis.negative_pct,
            neutral_pct=event_analysis.neutral_pct,
            top_themes=event_analysis.top_themes,
            representative_quotes=event_analysis.representative_quotes,
            llm_summary=event_analysis.llm_summary,
        )

    return AnalysisOutput(
        pre_sentiment=   pre_stats,
        review_analysis= review_result,
        event_analysis=  event_result,
        sentiment_delta= delta,
        top_topics=      review_topics,
        risk_signals=    signals,
        overall_risk=    overall_risk,
        gray_zone_count= gray_zone_count,
    )
