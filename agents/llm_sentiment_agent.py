# agents/llm_sentiment_agent.py
# LangGraph node: LLM re-scores VADER gray-zone reviews and updates
# the review sentiment stats in state.analysis.
#
# Only touches review_analysis — event_analysis is left untouched because
# event comments were already analysed by the LLM Map-Reduce pipeline.

from __future__ import annotations

import json

from agents.state import PipelineState, ReviewAnalysisResult, SentimentStats
from core.llm import chat as llm_chat
from config import LLM_MODEL_LIGHT, VADER_GRAY_LO

_SYSTEM = (
    "You are a sentiment analyser for Steam game reviews. "
    "For each numbered review, reply with exactly 'positive', 'neutral', or 'negative'. "
    'Return a JSON array of strings in the same order, e.g. ["positive","negative","neutral"].'
)


def llm_sentiment_node(state: PipelineState) -> dict:
    """
    Reads:  cleaned_reviews (post-update), analysis
    Writes: analysis (updated review_analysis sentiment), current_step

    Only reviews with VADER compound in the gray zone are re-scored.
    Updates analysis.review_analysis.sentiment — does NOT touch event_analysis.
    """
    df       = state.get("cleaned_reviews")
    analysis = state.get("analysis")

    if df is None or df.empty or analysis is None:
        return {"current_step": "llm_sentiment_done"}

    # Identify gray-zone rows
    gray_mask = df["vader_compound"].between(-VADER_GRAY_LO, VADER_GRAY_LO)
    gray_df   = df[gray_mask].head(100)

    if gray_df.empty:
        return {"current_step": "llm_sentiment_done"}

    numbered = "\n".join(
        f"[{i}] {row['review_content_processed'][:250]}"
        for i, (_, row) in enumerate(gray_df.iterrows())
    )

    try:
        raw = llm_chat(
            model=LLM_MODEL_LIGHT,
            system=_SYSTEM,
            user=numbered,
            max_tokens=1024,
            json_mode=True,
        )
        labels = json.loads(raw)
        if not isinstance(labels, list):
            raise ValueError("unexpected response format")
    except Exception as exc:
        return {
            "current_step": "llm_sentiment_done",
            "errors":       [f"ERROR: [llm_sentiment] {exc}"],
        }

    valid = {"positive", "neutral", "negative"}
    labels = [lbl if lbl in valid else "neutral" for lbl in labels]
    labels = (labels + ["neutral"] * len(gray_df))[:len(gray_df)]

    # Apply corrected labels to gray-zone rows
    import pandas as pd
    updated_df = df.copy()
    gray_indices = gray_df.index   # use index, not boolean mask
    n_labels = min(len(labels), len(gray_indices))
    updated_df.loc[gray_indices[:n_labels], "sentiment_label"] = labels[:n_labels]

    # Recompute review sentiment stats (NOT event sentiment)
    label_counts = updated_df["sentiment_label"].value_counts(normalize=True)
    corrected_stats = SentimentStats(
        mean_compound= float(updated_df["vader_compound"].mean()),
        positive_pct=  float(label_counts.get("positive", 0.0)),
        neutral_pct=   float(label_counts.get("neutral",  0.0)),
        negative_pct=  float(label_counts.get("negative", 0.0)),
        n_reviews=     len(updated_df),
    )

    # Update review_analysis + sentiment_delta, preserve event_analysis
    updated_review = None
    updated_delta = analysis.sentiment_delta
    if analysis.review_analysis is not None:
        updated_review = analysis.review_analysis.model_copy(
            update={"sentiment": corrected_stats}
        )
        # Recompute delta: post negative_pct - pre negative_pct
        updated_delta = round(
            corrected_stats.negative_pct - analysis.pre_sentiment.negative_pct, 4
        )

    updated_analysis = analysis.model_copy(
        update={
            "review_analysis": updated_review,
            "sentiment_delta": updated_delta,
        }
    )

    return {
        "cleaned_reviews": updated_df,
        "analysis":        updated_analysis,
        "current_step":    "llm_sentiment_done",
    }
