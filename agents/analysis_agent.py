# agents/analysis_agent.py
# LangGraph node: run sentiment comparison, topic tagging, and risk scoring.

from __future__ import annotations

from agents.state import PipelineState
from core.analysis import run_analysis
from core.features import extract_features


def analysis_node(state: PipelineState) -> dict:
    """
    Reads:  pre_reviews, cleaned_reviews (post-update)
    Writes: analysis, current_step

    pre_reviews also get feature extraction here so both windows
    are processed through the same NLP pipeline.
    """
    pre_df  = state.get("pre_reviews")
    post_df = state.get("cleaned_reviews")

    errors: list[str] = []

    if pre_df is None or pre_df.empty:
        errors.append("[analysis] pre_reviews is empty — baseline will be zero")
        import pandas as pd
        pre_df = pd.DataFrame()

    if post_df is None or post_df.empty:
        errors.append("[analysis] cleaned_reviews is empty — nothing to analyse")
        import pandas as pd
        post_df = pd.DataFrame()

    # Feature extraction on pre-update reviews (post already done in cleaning_node)
    if not pre_df.empty and "vader_compound" not in pre_df.columns:
        pre_df = extract_features(pre_df)

    analysis = run_analysis(pre_df, post_df)

    return {
        "analysis":     analysis,
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
