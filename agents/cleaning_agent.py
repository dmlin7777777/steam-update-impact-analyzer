# agents/cleaning_agent.py
# LangGraph node + conditional edge for post-update review cleaning.

from __future__ import annotations

from agents.state import CleaningOutput, PipelineState
from core.cleaning import run_cleaning, should_trigger_llm_review
from core.features import extract_features


def cleaning_node(state: PipelineState) -> dict:
    """
    Reads:  post_reviews
    Writes: cleaned_reviews, flagged_reviews, current_step

    Also runs feature extraction on cleaned_reviews so downstream
    agents have vader_compound, sentiment_label, etc. immediately.
    """
    df = state.get("post_reviews")

    if df is None or df.empty:
        return {
            "cleaned_reviews": df,
            "flagged_reviews": None,
            "current_step":   "cleaning_done",
            "errors":         ["WARN: [cleaning] post_reviews is empty — nothing to clean"],
        }

    cleaned, flagged = run_cleaning(df)

    # Add NLP features to cleaned reviews right away
    if not cleaned.empty:
        cleaned = extract_features(cleaned)

    total = len(cleaned) + len(flagged)
    output = CleaningOutput(
        n_cleaned=          len(cleaned),
        n_flagged=          len(flagged),
        flagged_ratio=      round(len(flagged) / total, 4) if total else 0.0,
        trigger_llm_review= should_trigger_llm_review(flagged, total),
    )

    return {
        "cleaned_reviews": cleaned,
        "flagged_reviews": flagged,
        "current_step":   "cleaning_done",
        # Surface summary in errors list only on warning-level issues
        **({"errors": [f"INFO: [cleaning] {output.n_flagged} rows flagged ({output.flagged_ratio:.1%})"]}
           if output.n_flagged > 0 else {}),
    }


def should_run_llm_review(state: PipelineState) -> str:
    """
    Conditional edge: route to llm_review if flagged ratio > threshold,
    otherwise go straight to analysis.
    """
    flagged = state.get("flagged_reviews")
    cleaned = state.get("cleaned_reviews")

    total = (len(cleaned) if cleaned is not None else 0) + \
            (len(flagged) if flagged is not None else 0)

    if flagged is not None and should_trigger_llm_review(flagged, total):
        return "llm_review"
    return "analysis"
