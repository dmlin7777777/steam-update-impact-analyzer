# core/analysis.py
# Post-update impact analysis: sentiment comparison, topic tagging, risk scoring.
#
# All functions are pure (no side effects). Topic tagging calls Claude haiku
# in batches to keep cost low.

from __future__ import annotations

import json
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

from agents.state import (
    AnalysisOutput,
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

_client = anthropic.Anthropic()


# ── Sentiment aggregation ─────────────────────────────────────────────────────

def compute_sentiment_stats(df: pd.DataFrame) -> SentimentStats:
    """
    Aggregate VADER scores into SentimentStats.
    Requires column: vader_compound, sentiment_label.
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


# ── Topic tagging (Claude haiku, batched) ─────────────────────────────────────

_TOPIC_SYSTEM = (
    "You are a game review analyst. Classify each review into exactly ONE topic "
    "from the list below. Reply with a JSON array — one label string per review, "
    "in the same order.\n\nTopics: " + ", ".join(TOPIC_LABELS)
)


def tag_topics(df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
    """
    Add a 'topic' column to df using Claude haiku batch classification.

    Sends reviews in batches of `batch_size` to reduce API calls.
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
            msg = _client.messages.create(
                model=LLM_MODEL_LIGHT,
                max_tokens=512,
                system=[{"type": "text", "text": _TOPIC_SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": numbered}],
            )
            raw = msg.content[0].text.strip()
            parsed: list[str] = json.loads(raw)
            # Clamp to known labels; fallback to 'other'
            parsed = [
                lbl if lbl in TOPIC_LABELS else "other"
                for lbl in parsed
            ]
            # Pad or truncate to match chunk length
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


# ── Risk scoring ──────────────────────────────────────────────────────────────

def compute_risk_signals(
    pre:  SentimentStats,
    post: SentimentStats,
    sentiment_delta: float,
) -> list[RiskSignal]:
    """
    Apply rule-based risk checks and return triggered signals.
    All thresholds come from config.ALERT_THRESHOLDS.
    """
    signals: list[RiskSignal] = []

    # R1 — Sentiment drop
    thresh_drop = ALERT_THRESHOLDS["sentiment_drop"]
    if sentiment_delta < -thresh_drop:
        signals.append(RiskSignal(
            rule="R1",
            severity=_drop_severity(sentiment_delta),
            value=round(sentiment_delta, 4),
            threshold=-thresh_drop,
            description=(
                f"Sentiment dropped {abs(sentiment_delta):.3f} points "
                f"(threshold {thresh_drop})"
            ),
        ))

    # R2 — Negative surge
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

    # R3 — Review-volume spike (approximated by n_reviews ratio)
    if pre.n_reviews > 0:
        rate_multiplier = post.n_reviews / pre.n_reviews
        thresh_rate     = ALERT_THRESHOLDS["review_rate_multiplier"]
        if rate_multiplier > thresh_rate:
            signals.append(RiskSignal(
                rule="R3",
                severity="MEDIUM",
                value=round(rate_multiplier, 2),
                threshold=thresh_rate,
                description=(
                    f"Review volume spiked {rate_multiplier:.1f}× baseline "
                    f"(threshold {thresh_rate:.0f}×)"
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


# ── Z-score anomaly check ─────────────────────────────────────────────────────

def compute_rolling_zscore(
    df: pd.DataFrame,
    window: str = "24h",
    col: str    = "vader_compound",
) -> pd.Series:
    """
    Rolling Z-score of `col` over a time window.
    df must have a DatetimeIndex or a 'timestamp' column.
    """
    s = df.set_index("timestamp")[col].sort_index()
    rolling = s.rolling(window, min_periods=2)
    z = (s - rolling.mean()) / rolling.std().replace(0, np.nan)
    return z.fillna(0.0)


# ── Orchestrating entry point ─────────────────────────────────────────────────

def run_analysis(
    pre_df:  pd.DataFrame,
    post_df: pd.DataFrame,
) -> AnalysisOutput:
    """
    Full analysis pipeline for one update event.

    Both DataFrames must already have feature columns from core/features.py.
    Topic tagging (Claude) is applied to post_df only.

    Returns an AnalysisOutput Pydantic model.
    """
    pre_stats  = compute_sentiment_stats(pre_df)
    post_stats = compute_sentiment_stats(post_df)
    delta      = round(post_stats.mean_compound - pre_stats.mean_compound, 4)

    # Topic tagging on post-update reviews
    post_tagged = tag_topics(post_df) if not post_df.empty else post_df
    top_topics  = aggregate_topics(post_tagged)

    # Risk signals
    signals      = compute_risk_signals(pre_stats, post_stats, delta)
    overall_risk = _overall_risk(signals)

    # Count ambiguous reviews that warrant LLM re-score
    gray_zone_count = int(
        post_df["vader_compound"]
        .between(-VADER_GRAY_LO, VADER_GRAY_LO)
        .sum()
    ) if not post_df.empty else 0

    return AnalysisOutput(
        pre_sentiment=   pre_stats,
        post_sentiment=  post_stats,
        sentiment_delta= delta,
        top_topics=      top_topics,
        risk_signals=    signals,
        overall_risk=    overall_risk,
        gray_zone_count= gray_zone_count,
    )
