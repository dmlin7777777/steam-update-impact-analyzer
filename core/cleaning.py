# core/cleaning.py
# Column names depend on the actual DataFrame passed in.
# Primary text column assumed: "review_text"

import re

import numpy as np
import pandas as pd

from config import LLM_REVIEW_THRESHOLD


def _is_repeated_chars(text: str) -> bool:
    """Return True if the text consists entirely of one repeated character (e.g. 'aaaaaa')."""
    return bool(re.fullmatch(r'(.)\1+', text.strip()))


def _special_char_ratio(text: str) -> float:
    """Fraction of characters that are not word chars, whitespace, or CJK."""
    special = len(re.findall(r'[^\w\s一-鿿]', text))
    return special / max(len(text), 1)


def run_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a review DataFrame.

    Hard deletes (rows discarded entirely, not sent anywhere):
    - review_text is null or empty

    Flagged rows (returned in flagged_df for LLM review, excluded from cleaned_df):
    - text length < 10 characters
    - text consists entirely of one repeated character (e.g. "aaaaaa")
    - special character ratio > 40%

    Note: exact duplicate detection is left to the caller; this function does not
    deduplicate rows so that flagged_df + cleaned_df together account for every
    non-null input row.

    Returns:
        cleaned_df  – non-null, non-flagged rows (may still contain duplicates)
        flagged_df  – suspicious rows for LLM review
    """
    # --- hard delete: null / empty text ---
    mask_null = df["review_text"].isna() | (df["review_text"].astype(str).str.strip() == "")
    df = df[~mask_null].copy()

    # --- flag suspicious rows ---
    col = df["review_text"].astype(str)

    short_text     = col.str.strip().str.len() < 10
    repeated_chars = col.str.strip().apply(_is_repeated_chars)
    high_special   = col.apply(_special_char_ratio) > 0.40

    flag_mask = short_text | repeated_chars | high_special

    flagged_df = df[flag_mask].copy()
    cleaned_df = df[~flag_mask].copy()

    return cleaned_df, flagged_df


def should_trigger_llm_review(flagged_df: pd.DataFrame, total: int) -> bool:
    """Return True when flagged rows exceed LLM_REVIEW_THRESHOLD of the original total."""
    if total == 0:
        return False
    return (len(flagged_df) / total) > LLM_REVIEW_THRESHOLD
