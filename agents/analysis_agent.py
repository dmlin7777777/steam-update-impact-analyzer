# agents/analysis_agent.py
# LangGraph node: run sentiment comparison, topic tagging, and risk scoring.

from __future__ import annotations

import pandas as pd

from agents.state import PipelineState
from core.analysis import run_analysis
from core.features import extract_features


def analysis_node(state: PipelineState) -> dict:
    """
    Reads:  pre_reviews, cleaned_reviews (post-update), event_comments, patch_notes
    Writes: analysis, pre_reviews (enriched), current_step

    Two independent analysis paths
    ------------------------------
    1. Event comments (primary):
       Raw comment text -> Map-Reduce LLM pipeline (core/event_analysis.py).
       Every comment is read by the LLM directly. No VADER, no preprocessing.

    2. Reviews (secondary):
       VADER sentiment + voted_up disagreement detection + distribution summary.
       Topic tagging via Claude Haiku.

    Both paths' results are passed to run_analysis() which computes risk signals
    from both sources independently.
    """
    pre_df   = state.get("pre_reviews")
    post_df  = state.get("cleaned_reviews")
    event_df = state.get("event_comments")
    patch_notes = state.get("patch_notes", [])

    errors: list[str] = []

    if pre_df is None or pre_df.empty:
        errors.append("WARN: [analysis] pre_reviews is empty -- baseline will be zero")
        pre_df = pd.DataFrame()

    if post_df is None or post_df.empty:
        errors.append("WARN: [analysis] cleaned_reviews is empty -- nothing to analyse")
        post_df = pd.DataFrame()

    # ── Feature extraction on reviews ────────────────────────────────────────
    if not pre_df.empty and "vader_compound" not in pre_df.columns:
        pre_df = extract_features(pre_df)

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

    # ── Event comment analysis (Map-Reduce LLM path) ────────────────────────
    event_analysis_result = None
    if event_df is not None and not event_df.empty:
        from core.event_analysis import analyse_event_comments

        # Extract raw comment texts — no preprocessing
        comment_texts = event_df["review_content"].dropna().tolist()

        # Build patch notes context
        patch_context = ""
        if patch_notes:
            patch_context = "\n\n".join(
                f"--- {pn.title} ---\n{pn.contents[:600]}"
                for pn in patch_notes[:3]
            )

        if comment_texts:
            errors.append(
                f"INFO: [analysis] Analysing {len(comment_texts)} announcement comments "
                f"via LLM Map-Reduce (primary signal)."
            )
            event_analysis_result = analyse_event_comments(
                comment_texts, patch_context
            )
            if event_analysis_result:
                errors.append(
                    f"INFO: [analysis] Announcement sentiment: "
                    f"{event_analysis_result.positive_pct:.0%} pos / "
                    f"{event_analysis_result.negative_pct:.0%} neg / "
                    f"{event_analysis_result.neutral_pct:.0%} neu "
                    f"({event_analysis_result.n_comments} comments)"
                )

    # ── Run analysis (two independent paths) ─────────────────────────────────
    analysis = run_analysis(pre_df, post_df, event_analysis_result)

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
