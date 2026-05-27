# core/scraper.py
# Steam API client: reviews, patch notes, and app metadata.
#
# Three public functions, each returns plain Python types (no side effects):
#   fetch_app_details(appid)              → dict
#   fetch_reviews(appid, start, end)      → pd.DataFrame
#   fetch_patch_notes(appid, since, n)    → list[dict]
#
# DataFrame schema produced by fetch_reviews:
#   review_id         str   Steam recommendationid (or surrogate for cached rows)
#   review_content    str   raw review text (normalised column name)
#   voted_up          bool  recommended or not
#   timestamp         datetime (UTC-aware)
#   playtime_hours    float hours played at review time
#   votes_up          int   helpful votes
#   votes_funny       int   funny votes
#
# Data-source priority
# --------------------
# 1. Local SQLite cache (core/local_cache.py) — populated offline by
#    scripts/import_cache.py.  Covers any historical window that was
#    imported.  Zero network cost, instant.
# 2. Live Steam API — cursor-paginated, newest-first.  Works well for
#    updates within the last ~30 days on high-traffic games.

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config import (
    MAX_REVIEWS_TOTAL,
    REVIEWS_PER_PAGE,
    SCRAPER_DELAY_SEC,
    SCRAPER_RETRIES,
    STEAM_APP_DETAILS_URL,
    STEAM_NEWS_URL,
    STEAM_REVIEWS_URL,
)


# ── Exceptions ────────────────────────────────────────────────────────────────

class SteamAPIError(Exception):
    """Raised when the Steam API returns an error or the request fails."""


# ── Internal HTTP helper ──────────────────────────────────────────────────────

def _get(url: str, params: dict) -> dict:
    """GET with exponential-backoff retry. Returns parsed JSON."""
    for attempt in range(SCRAPER_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == SCRAPER_RETRIES - 1:
                raise SteamAPIError(f"Steam API unreachable after {SCRAPER_RETRIES} attempts: {exc}") from exc
            time.sleep(1.5 ** attempt)   # 1 s, 1.5 s, 2.25 s …
    return {}


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_app_details(appid: str) -> dict:
    """
    Return basic game metadata from the Steam store API.

    Returns:
        dict with keys: name, developer, header_image
        Falls back to safe defaults if the app is not found.
    """
    data = _get(STEAM_APP_DETAILS_URL, {"appids": appid, "filters": "basic"})
    entry = data.get(str(appid), {})
    if not entry.get("success"):
        return {"name": f"App {appid}", "developer": "Unknown", "header_image": ""}

    d = entry["data"]
    developers = d.get("developers") or ["Unknown"]
    return {
        "name":         d.get("name", f"App {appid}"),
        "developer":    developers[0],
        "header_image": d.get("header_image", ""),
    }


def fetch_reviews(
    appid: str,
    start_dt: datetime,
    end_dt: datetime,
    max_reviews: int = MAX_REVIEWS_TOTAL,
) -> pd.DataFrame:
    """
    Fetch English reviews posted within [start_dt, end_dt].

    Data-source priority:
      1. Local SQLite cache — if the requested window is fully covered by
         cached data, returns instantly with no network call.
      2. Live Steam API — cursor-paginated, newest-first.

    Historical-data limitation (live API only): Steam returns reviews
    newest-first with no date-range parameter.  For very active games
    (CS2, Dota 2 …) windows more than ~30 days in the past require paging
    through huge volumes; max_reviews acts as the budget.

    Args:
        appid:       Steam App ID string.
        start_dt:    Window start (UTC-aware, inclusive).
        end_dt:      Window end   (UTC-aware, inclusive).
        max_reviews: Hard cap on rows scanned by the live API (not cached rows).

    Returns:
        DataFrame with columns defined in module docstring.
        Empty DataFrame (correct columns) if no reviews found.
    """
    # ── 1. Try local cache first ──────────────────────────────────────────────
    try:
        from core.local_cache import query_reviews as _cache_query, cache_date_range
        cached_range = cache_date_range(appid)
        if cached_range is not None:
            cache_min, cache_max = cached_range
            _start = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
            _end   = end_dt   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
            # Use cache when requested window is fully within cached range
            if cache_min <= _start and _end <= cache_max:
                return _cache_query(appid, _start, _end)
    except Exception:
        pass  # cache unavailable — fall through to live API
    url    = STEAM_REVIEWS_URL.format(appid=appid)
    params = {
        "json":          1,
        "filter":        "recent",
        "language":      "english",
        "num_per_page":  REVIEWS_PER_PAGE,
        "purchase_type": "all",
        "cursor":        "*",
    }

    # Ensure timestamps are UTC-aware for comparison
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    rows: list[dict] = []
    scanned: int = 0          # total reviews examined (including skipped future ones)

    while scanned < max_reviews:
        data = _get(url, params)

        if data.get("success") != 1:
            break

        batch = data.get("reviews") or []
        if not batch:
            break

        reached_before_window = False
        for rev in batch:
            scanned += 1
            ts = datetime.fromtimestamp(rev["timestamp_created"], tz=timezone.utc)

            if ts < start_dt:
                # Reviews sorted newest-first; anything older is out of range.
                reached_before_window = True
                break

            if ts <= end_dt:
                rows.append({
                    "review_id":      rev["recommendationid"],
                    "review_content": rev["review"],
                    "voted_up":       bool(rev["voted_up"]),
                    "timestamp":      ts,
                    "playtime_hours": round(
                        rev.get("author", {}).get("playtime_forever", 0) / 60, 1
                    ),
                    "votes_up":    rev.get("votes_up", 0),
                    "votes_funny": rev.get("votes_funny", 0),
                })
            # else: review is in the future relative to end_dt — skip, keep scanning

        if reached_before_window:
            break

        # Advance cursor for next page
        next_cursor = data.get("cursor", "")
        if not next_cursor or next_cursor == params["cursor"]:
            break
        params["cursor"] = next_cursor

        time.sleep(SCRAPER_DELAY_SEC)

    return _to_dataframe(rows)


def fetch_patch_notes(
    appid: str,
    since_dt: datetime,
    count: int = 20,
) -> list[dict]:
    """
    Fetch news items that look like patch notes / update announcements.

    Filters by publication date (>= since_dt) and by title keywords / Steam tags.

    Args:
        appid:    Steam App ID string.
        since_dt: Only return items published on or after this datetime.
        count:    How many news items to request from the API.

    Returns:
        List of dicts with keys: gid, title, url, published_at, contents.
        Sorted newest-first.
    """
    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=timezone.utc)

    data = _get(
        STEAM_NEWS_URL,
        {"appid": appid, "count": count, "maxlength": 3000, "format": "json"},
    )

    items = (data.get("appnews") or {}).get("newsitems") or []

    _UPDATE_KEYWORDS = {"patch", "update", "fix", "hotfix", "changelog", "maintenance"}

    result: list[dict] = []
    for item in items:
        published_at = datetime.fromtimestamp(item["date"], tz=timezone.utc)
        if published_at < since_dt:
            continue

        # Identify patch notes by Steam tags or title keywords.
        # Tags field is a plain list[str], e.g. ["patchnotes"].
        raw_tags    = item.get("tags") or []
        tags        = {
            (t.get("tag", "") if isinstance(t, dict) else str(t)).lower()
            for t in raw_tags
        }
        title_words = set(item.get("title", "").lower().split())

        is_update = bool(
            tags & {"patchnotes", "patch_notes"}
            or title_words & _UPDATE_KEYWORDS
        )

        if is_update:
            result.append({
                "gid":          str(item["gid"]),
                "title":        item.get("title", ""),
                "url":          item.get("url", ""),
                "published_at": published_at,
                "contents":     (item.get("contents") or "")[:2000],
            })

    # Newest-first
    result.sort(key=lambda x: x["published_at"], reverse=True)
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

_EMPTY_COLUMNS = [
    "review_id", "review_content", "voted_up",
    "timestamp", "playtime_hours", "votes_up", "votes_funny",
]

def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
