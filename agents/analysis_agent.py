# agents/analysis_agent.py
# LangGraph node: run sentiment comparison, topic tagging, and risk scoring.

from __future__ import annotations

import pandas as pd

from agents.state import PipelineState
from core.analysis import run_analysis
from core.features import extract_features


def analysis_node(state: PipelineState) -> dict:
    """
    Reads:  pre_reviews, cleaned_reviews (post-update), event_comments
    Writes: analysis, pre_reviews (enriched), current_step

    Data architecture
    -----------------
    Event comments and reviews are kept SEPARATE and passed independently to
    run_analysis().  Inside run_analysis(), sentiment is computed for each
    source independently, then blended at a fixed ratio:

        post_sentiment = EVENT_BLEND_RATIO  x event_comment_sentiment   (primary)
                       + (1-EVENT_BLEND_RATIO) x review_sentiment       (secondary)

    This guarantees that event comments always dominate regardless of how many
    reviews exist in the window.  Topic tagging still uses all data combined.
    """
    pre_df   = state.get("pre_reviews")
    post_df  = state.get("cleaned_reviews")
    event_df = state.get("event_comments")

    errors: list[str] = []

    if pre_df is None or pre_df.empty:
        errors.append("[analysis] pre_reviews is empty — baseline will be zero")
        pre_df = pd.DataFrame()

    if post_df is None or post_df.empty:
        errors.append("[analysis] cleaned_reviews is empty — nothing to analyse")
        post_df = pd.DataFrame()

    # Feature extraction on pre-update reviews
    if not pre_df.empty and "vader_compound" not in pre_df.columns:
        pre_df = extract_features(pre_df)

    # Feature extraction on post-update reviews (only rows missing features)
    if not post_df.empty:
        missing = (
            post_df["vader_compound"].isna()
            if "vader_compound" in post_df.columns
            else pd.Series(True, index=post_df.index)
        )
        if missing.any():
            has_feat   = post_df[~missing]
            needs_feat = extract_features(post_df[missing].copy())
            post_df    = pd.concat([has_feat, needs_feat], ignore_index=True)

    # event_df feature extraction is handled inside run_analysis()

    if event_df is not None and not event_df.empty:
        errors.append(
            f"[analysis] {len(event_df)} announcement comments available "
            f"(primary signal, 70% blend weight)."
        )

    analysis = run_analysis(pre_df, post_df, event_df)

    return {
        "analysis":     analysis,
        "pre_reviews":  pre_df,
        "current_step": "analysis_done",
        **({"errors": errors} if errors else {}),
    }


def should_run_llm_sentiment(state: PipelineState) -> str:
    """
    Conditional edge: if there are gray-zone reviews, send them to Claude
    for a more nuanced sentiment re-score; otherwise go to recommendation.
    """
    analysis = state.get("analysis")
    if analysis and analysis.gray_zone_count > 0:
        return "llm_sentiment"
    return "recommendation"
