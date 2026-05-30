# core/cleaning.py
# Review data cleaning: hard-exclude low-quality / non-analysable rows.
#
# Design rationale (data-driven, 59k CS2 reviews)
# ------------------------------------------------
# Steam enforces ~20 char minimum review length, so the old <10 char rule
# had a 0% hit rate. Rules are now split into two tiers:
#
#   Hard exclude (non-analysable — VADER/LLM cannot process):
#     - Null / empty content
#     - Non-English text (Cyrillic/CJK/Arabic/Thai dominant)   ~2.8%
#     - Punctuation-only / no alphabetic content                ~4.3%
#
#   Flagged exclude (suspicious — literature-backed spam/bot signals):
#     Behavioral:
#       - playtime_at_review < 2h AND num_reviews > 5           ~0.13%
#       - playtime_at_review < 30min                            ~0.43%
#       - num_reviews > 50 AND playtime_at_review < 10h         ~0.09%
#     Content:
#       - Repetitive words (unique ratio < 30%)                 ~0.12%
#       - Multiple URLs (>= 2)                                  ~0.34%
#       - Repeated character >= 10 times                        ~0.95%
#
# Total exclude rate: ~9% (dominated by non-English leak-through).
# No LLM call in the cleaning path — rules are deterministic.
#
# References:
#   - "Spam Review Detection Techniques: A Systematic Literature Review"
#     (Applied Sciences, 2019) — behavioral > content features
#   - "Detecting Spam Game Reviews on Steam with a Semi-Supervised Approach"
#     (ANU/FDG 2021) — 15% spam rate in Steam reviews
#   - "Effective Opinion Spam Detection: Metadata vs Content"
#     (JDIS, 2020) — metadata features more discriminative

import re
from typing import Optional

import pandas as pd


# ── Non-English / non-analysable detection ───────────────────────────────────

_CYRILLIC_RE = re.compile(r'[Ѐ-ӿ]')
_CJK_RE      = re.compile(r'[一-鿿]')
_ARABIC_RE   = re.compile(r'[؀-ۿ]')
_THAI_RE     = re.compile(r'[฀-๿]')
_ALPHA_RE    = re.compile(r'[a-zA-Z]')


def _is_non_english(text: str) -> bool:
    """True if text is dominated by non-Latin script or has no alphabetic chars."""
    stripped = text.strip()
    if not stripped:
        return True
    alpha_count = len(_ALPHA_RE.findall(stripped))
    alpha_ratio = alpha_count / len(stripped)
    if alpha_ratio < 0.2:
        # Check if it's actual non-English script vs pure punctuation
        has_script = bool(
            _CYRILLIC_RE.search(stripped) or _CJK_RE.search(stripped) or
            _ARABIC_RE.search(stripped) or _THAI_RE.search(stripped)
        )
        if has_script:
            return True
    return False


def _is_punctuation_only(text: str) -> bool:
    """True if text has no word characters (letters/digits)."""
    return bool(re.fullmatch(r'[\W\s]+', text.strip()))


# ── Content quality flags ────────────────────────────────────────────────────

def _has_repeated_char(text: str, threshold: int = 10) -> bool:
    """True if any character repeats >= threshold times consecutively."""
    return bool(re.search(rf'(.)\1{{{threshold - 1},}}', text))


def _is_repetitive_words(text: str, threshold: float = 0.3) -> bool:
    """True if unique-word ratio is below threshold (copy-paste spam)."""
    words = text.lower().split()
    if len(words) < 4:
        return False
    return len(set(words)) / len(words) < threshold


def _has_multiple_urls(text: str, threshold: int = 2) -> bool:
    """True if text contains >= threshold URLs (promotional spam)."""
    return len(re.findall(r'https?://', text)) >= threshold


# ── Behavioral flags ────────────────────────────────────────────────────────

def _flag_behavioral(df: pd.DataFrame) -> pd.Series:
    """
    Return boolean mask of rows that match behavioral spam/bot signals.
    Requires columns: playtime_at_review (minutes), num_reviews.
    Gracefully returns all-False if columns are missing.
    """
    mask = pd.Series(False, index=df.index)

    has_pt = "playtime_at_review" in df.columns
    has_nr = "num_reviews" in df.columns

    if has_pt:
        pt_hours = df["playtime_at_review"] / 60

        # B1: Low playtime + prolific reviewer = bot signal
        if has_nr:
            mask = mask | ((pt_hours < 2) & (df["num_reviews"] > 5))

        # B3: Extremely low playtime (< 30 min)
        mask = mask | (pt_hours < 0.5)

        # B4: Prolific reviewer + low playtime = bot farm
        if has_nr:
            mask = mask | ((df["num_reviews"] > 50) & (pt_hours < 10))

    return mask


# ── Public API ──────────────────────────────────────────────────────────────

def run_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a review DataFrame using data-driven exclusion rules.

    Two-tier exclusion (no LLM call — all rules are deterministic):

    Hard exclude (non-analysable):
      - review_content is null or empty
      - Non-English text (Cyrillic, CJK, Arabic, Thai dominant)
      - Punctuation-only (no alphabetic content)

    Flagged exclude (spam/bot signals):
      Behavioral: low playtime + high review count, extremely low playtime,
                  prolific reviewer + low playtime
      Content:    repeated characters, repetitive words, multiple URLs

    Returns:
        cleaned_df  – rows suitable for analysis
        excluded_df – rows excluded with 'exclude_reason' column
    """
    if df.empty:
        empty = df.copy()
        empty["exclude_reason"] = pd.Series(dtype=str)
        return empty, empty

    # Work on a copy
    df = df.copy()
    col = df["review_content"].astype(str)
    reasons = pd.Series("", index=df.index)

    # ── Hard excludes ────────────────────────────────────────────────────────
    mask_null = df["review_content"].isna() | (col.str.strip() == "")
    reasons = reasons.where(~mask_null, "null_or_empty")

    mask_punct = (~mask_null) & col.apply(_is_punctuation_only)
    reasons = reasons.where(~mask_punct, "punctuation_only")

    mask_nonenglish = (~mask_null) & (~mask_punct) & col.apply(_is_non_english)
    reasons = reasons.where(~mask_nonenglish, "non_english")

    # ── Content flags ────────────────────────────────────────────────────────
    remaining = ~(mask_null | mask_punct | mask_nonenglish)

    mask_repeated = remaining & col.apply(_has_repeated_char)
    reasons = reasons.where(~mask_repeated, "repeated_char")

    mask_repetitive = remaining & col.apply(_is_repetitive_words)
    reasons = reasons.where(~mask_repetitive, "repetitive_words")

    mask_urls = remaining & col.apply(_has_multiple_urls)
    reasons = reasons.where(~mask_urls, "multiple_urls")

    # ── Behavioral flags ─────────────────────────────────────────────────────
    mask_behavioral = remaining & _flag_behavioral(df)
    reasons = reasons.where(~mask_behavioral, "behavioral_spam")

    # ── Split ────────────────────────────────────────────────────────────────
    exclude_mask = reasons != ""
    excluded_df = df[exclude_mask].copy()
    excluded_df["exclude_reason"] = reasons[exclude_mask]

    cleaned_df = df[~exclude_mask].copy()

    return cleaned_df, excluded_df
