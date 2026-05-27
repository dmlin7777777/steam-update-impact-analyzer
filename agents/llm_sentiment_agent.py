# agents/llm_sentiment_agent.py
# LangGraph node: Claude haiku re-scores VADER gray-zone reviews and updates
# the sentiment stats in state.analysis.

from __future__ import annotations

import json

import anthropic

from agents.state import PipelineState, SentimentStats
from config import LLM_MODEL_LIGHT, VADER_GRAY_LO

_client = anthropic.Anthropic()

_SYSTEM = (
    "You are a sentiment analyser for Steam game reviews. "
    "For each numbered review, reply with exactly 'positive', 'neutral', or 'negative'. "
    'Return a JSON array of strings in the same order, e.g. ["positive","negative","neutral"].'
)


def llm_sentiment_node(state: PipelineState) -> dict:
    """
    Reads:  cleaned_reviews (post-update), analysis
    Writes: analysis (updated sentiment stats), current_step

    Only reviews with VADER compound in the gray zone are re-scored.
    The corrected labels are used to recompute SentimentStats, which
    replaces analysis.post_sentiment.
    """
    df       = state.get("cleaned_reviews")
    analysis = state.get("analysis")

    if df is None or df.empty or analysis is None:
        return {"current_step": "llm_sentiment_done"}

    # Identify gray-zone rows
    gray_mask = df["vader_compound"].between(-VADER_GRAY_LO, VADER_GRAY_LO)
    gray_df   = df[gray_mask].head(100)   # cap to avoid runaway cost

    if gray_df.empty:
        return {"current_step": "llm_sentiment_done"}

    numbered = "\n".join(
        f"[{i}] {row['review_content_processed'][:250]}"
        for i, (_, row) in enumerate(gray_df.iterrows())
    )

    try:
        msg = _client.messages.create(
            model=LLM_MODEL_LIGHT,
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": numbered}],
        )
        raw     = msg.content[0].text.strip()
        labels  = json.loads(raw)   # list[str]
        if not isinstance(labels, list):
            raise ValueError("unexpected response format")
    except Exception as exc:
        return {
            "current_step": "llm_sentiment_done",
            "errors":       [f"[llm_sentiment] {exc}"],
        }

    # Clamp to valid labels
    valid = {"positive", "neutral", "negative"}
    labels = [lbl if lbl in valid else "neutral" for lbl in labels]
    # Pad/truncate to match gray_df length
    labels = (labels + ["neutral"] * len(gray_df))[: len(gray_df)]

    # Apply corrected labels to a copy of the df
    import pandas as pd
    updated_df = df.copy()
    updated_df.loc[gray_mask.values[:len(updated_df)], "sentiment_label"] = (
        labels[:gray_mask.sum()]
    )

    # Recompute SentimentStats from corrected labels
    label_counts = updated_df["sentiment_label"].value_counts(normalize=True)
    corrected_stats = SentimentStats(
        mean_compound= float(updated_df["vader_compound"].mean()),
        positive_pct=  float(label_counts.get("positive", 0.0)),
        neutral_pct=   float(label_counts.get("neutral",  0.0)),
        negative_pct=  float(label_counts.get("negative", 0.0)),
        n_reviews=     len(updated_df),
    )

    # Rebuild analysis with corrected post_sentiment
    updated_analysis = analysis.model_copy(
        update={"post_sentiment": corrected_stats}
    )

    return {
        "cleaned_reviews": updated_df,
        "analysis":        updated_analysis,
        "current_step":    "llm_sentiment_done",
    }
