# agents/cleaning_agent.py
# LangGraph node: remove null/empty reviews + run feature extraction.
#
# Why no LLM review step?
# -----------------------
# Steam enforces a minimum review length (~20 chars at platform level).
# The cleaning rules (short text <10 chars, repeated chars, high special-char
# ratio) have a verified 0% trigger rate across 71k real Steam reviews.
# The former llm_review_node conditional branch was removed as dead code.

from __future__ import annotations

from agents.state import PipelineState
from core.cleaning import run_cleaning
from core.features import extract_features


def cleaning_node(state: PipelineState) -> dict:
    """
    Reads:  post_reviews
    Writes: cleaned_reviews, current_step

    Removes null/empty reviews and runs feature extraction so downstream
    agents have vader_compound, sentiment_label, etc. immediately.
    """
    df = state.get("post_reviews")

    if df is None or df.empty:
        return {
            "cleaned_reviews": df,
            "current_step":   "cleaning_done",
            "errors":         ["WARN: [cleaning] post_reviews is empty — nothing to clean"],
        }

    cleaned, flagged = run_cleaning(df)

    # Add NLP features to cleaned reviews right away
    if not cleaned.empty:
        cleaned = extract_features(cleaned)

    return {
        "cleaned_reviews": cleaned,
        "current_step":   "cleaning_done",
        **({"errors": [f"INFO: [cleaning] {len(flagged)} rows flagged and excluded"]}
           if len(flagged) > 0 else {}),
    }
