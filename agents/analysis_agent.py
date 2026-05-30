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
    Writes: analysis, current_step

    Event comments (from the update announcement page) are merged into the
    post-update DataFrame before analysis.  They are treated as high-signal
    post-update data since they are anchored to the specific update.

    pre_reviews also get feature extraction here so both windows are processed
    through the same NLP pipeline.
    """
    pre_df      = state.get("pre_reviews")
    post_df     = state.get("cleaned_reviews")
    event_df    = state.get("event_comments")

    errors: list[str] = []

    if pre_df is None or pre_df.empty:
        errors.append("[analysis] pre_reviews is empty — baseline will be zero")
        pre_df = pd.DataFrame()

    if post_df is None or post_df.empty:
        errors.append("[analysis] cleaned_reviews is empty — nothing to analyse")
        post_df = pd.DataFrame()

    # Merge event comments into post-update window
    # Event comments carry higher signal: they are direct reactions to THIS update.
    # We append them so they participate in sentiment stats and topic tagging.
    post_df = _merge_event_comments(post_df, event_df, errors)

    # Feature extraction on pre-update reviews (post already done in cleaning_node,
    # except for any newly appended event-comment rows)
    if not pre_df.empty and "vader_compound" not in pre_df.columns:
        pre_df = extract_features(pre_df)

    if not post_df.empty:
        # Re-run features on rows that may be missing them (e.g. event comments)
        missing_feat = post_df["vader_compound"].isna() if "vader_compound" in post_df.columns \
                       else pd.Series(True, index=post_df.index)
        if missing_feat.any():
            # Avoid redundant computation: only process new rows
            needs_feat = post_df[missing_feat].copy()
            has_feat   = post_df[~missing_feat]
            needs_feat = extract_features(needs_feat)
            post_df    = pd.concat([has_feat, needs_feat], ignore_index=True)

    analysis = run_analysis(pre_df, post_df)

    return {
        "analysis":     analysis,
        "pre_reviews":  pre_df,    # return enriched version (now has vader_compound)
        "current_step": "analysis_done",
        **({"errors": errors} if errors else {}),
    }


def _merge_event_comments(
    post_df: pd.DataFrame,
    event_df: pd.DataFrame | None,
    errors: list[str],
) -> pd.DataFrame:
    """
    Append event_comments to post_df.
    Normalises the event_df columns to match the review schema before concat.
    """
    if event_df is None or event_df.empty:
        return post_df

    # Ensure event_df has the columns post_df expects
    needed = ["review_id", "review_content", "voted_up",
              "timestamp", "playtime_hours", "votes_up", "votes_funny"]
    missing_cols = [c for c in needed if c not in event_df.columns]
    if missing_cols:
        errors.append(
            f"[analysis] event_comments missing columns {missing_cols}; skipping merge."
        )
        return post_df

    # Stamp weight=1.0 on regular reviews so both sides have the column
    post_df = post_df.copy()
    if "weight" not in post_df.columns:
        post_df["weight"] = 1.0
    if "source" not in post_df.columns:
        post_df["source"] = "review"

    keep_cols = needed + [
        c for c in ["source", "weight"] if c in event_df.columns
    ]
    ec_subset = event_df[keep_cols].copy()
    if "source" not in ec_subset.columns:
        ec_subset["source"] = "event_comment"
    if "weight" not in ec_subset.columns:
        ec_subset["weight"] = 1.0   # fallback if old dataframe without weight column

    ec_w = ec_subset["weight"].iloc[0] if not ec_subset.empty else 1.0
    errors.append(
        f"[analysis] Merged {len(ec_subset)} event comments "
        f"(weight={ec_w:.1f}x) into post-update window."
    )
    return pd.concat([post_df, ec_subset], ignore_index=True)


def should_run_llm_sentiment(state: PipelineState) -> str:
    """
    Conditional edge: if there are gray-zone reviews, send them to Claude
    for a more nuanced sentiment re-score; otherwise go to recommendation.
    """
    analysis = state.get("analysis")
    if analysis and analysis.gray_zone_count > 0:
        return "llm_sentiment"
    return "recommendation"
