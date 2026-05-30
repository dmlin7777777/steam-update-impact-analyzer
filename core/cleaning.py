# core/cleaning.py
# Primary text column: "review_content" (matches actual Steam review DataFrames)

import re

import pandas as pd


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
    - review_content is null or empty

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
    mask_null = df["review_content"].isna() | (df["review_content"].astype(str).str.strip() == "")
    df = df[~mask_null].copy()

    # --- flag suspicious rows ---
    col = df["review_content"].astype(str)

    short_text     = col.str.strip().str.len() < 10
    repeated_chars = col.str.strip().apply(_is_repeated_chars)
    high_special   = col.apply(_special_char_ratio) > 0.40

    flag_mask = short_text | repeated_chars | high_special

    flagged_df = df[flag_mask].copy()
    cleaned_df = df[~flag_mask].copy()

    return cleaned_df, flagged_df


