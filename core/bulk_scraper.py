# core/bulk_scraper.py
# One-time bulk review scrape for new games.
#
# Problem
# -------
# Steam's review API is newest-first only.  For recent updates (< ~30 days)
# the live scraper works fine.  For historical analysis you need a local cache
# populated by a bulk scrape.  Running a bulk scrape at game-add time gives the
# system a ~30-day look-back window immediately (for CS2-class traffic games)
# and a much larger window for lower-traffic games.
#
# Design
# ------
# * Paginates filter=recent just like the old scrapper_optimized.py, but writes
#   directly to the SQLite cache (core/local_cache.py) instead of a CSV.
# * Reports progress via a callback so the UI can show a live count.
# * max_reviews defaults to 60_000 (same ceiling as the old scraper).
# * Runs synchronously; callers that want non-blocking behaviour should use
#   threading.Thread or concurrent.futures.

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import requests

from config import REVIEWS_PER_PAGE, SCRAPER_DELAY_SEC, SCRAPER_RETRIES

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_EMPTY_COLS = [
    "review_id", "review_content", "voted_up",
    "timestamp", "playtime_hours", "votes_up", "votes_funny",
]


def bulk_scrape(
    appid: str,
    max_reviews: Optional[int] = 60_000,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    store_in_cache: bool = True,
) -> pd.DataFrame:
    """
    Scrape up to *max_reviews* most-recent English reviews for *appid* and
    optionally write them to the local SQLite cache.

    Args:
        appid:         Steam App ID string.
        max_reviews:   Maximum rows to fetch (hard cap).  Pass None to scrape
                       until the Steam cursor is exhausted (all reviews).
        progress_cb:   Optional callback(fetched, total_so_far) called after
                       each page.  Use for progress bars / spinners.
        store_in_cache: If True (default), insert rows into the local cache.

    Returns:
        DataFrame with canonical review columns (review_id, review_content …).
        Suitable for passing directly to extract_features().
    """
    url    = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        "json":          1,
        "filter":        "recent",
        "language":      "english",
        "num_per_page":  REVIEWS_PER_PAGE,
        "purchase_type": "all",
        "cursor":        "*",
    }

    rows: list[dict] = []
    cursor = "*"

    while max_reviews is None or len(rows) < max_reviews:
        params["cursor"] = cursor
        data = _get(url, params)

        if data.get("success") != 1:
            break

        batch = data.get("reviews") or []
        if not batch:
            break

        for rev in batch:
            ts = datetime.fromtimestamp(rev["timestamp_created"], tz=timezone.utc)
            pa = rev.get("author", {}).get("playtime_at_review", 0) or 0
            ph = rev.get("author", {}).get("playtime_forever", 0) or 0
            rows.append({
                "review_id":      str(rev["recommendationid"]),
                "review_content": rev["review"],
                "voted_up":       bool(rev["voted_up"]),
                "timestamp":      ts,
                "playtime_hours": round((pa or ph) / 60, 1),
                "votes_up":       rev.get("votes_up", 0),
                "votes_funny":    rev.get("votes_funny", 0),
            })

        if progress_cb:
            progress_cb(len(batch), len(rows))

        if max_reviews is not None and len(rows) >= max_reviews:
            rows = rows[:max_reviews]
            break

        # Advance cursor
        next_cursor = data.get("cursor", "")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

        time.sleep(SCRAPER_DELAY_SEC)

    df = _to_df(rows)

    if store_in_cache and not df.empty:
        from core.local_cache import import_dataframe
        import_dataframe(df, appid, legacy=False)

    return df


# ── Internal HTTP helper ──────────────────────────────────────────────────────

def _get(url: str, params: dict) -> dict:
    for attempt in range(SCRAPER_RETRIES):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt < SCRAPER_RETRIES - 1:
                time.sleep(1.5 ** attempt)
    return {}


def _to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
