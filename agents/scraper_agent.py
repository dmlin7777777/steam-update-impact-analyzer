# agents/scraper_agent.py
# LangGraph node: fetch Steam reviews + patch notes + event comments for an
# update event.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from agents.state import PatchNote, PipelineState, ScraperOutput
from core.scraper import SteamAPIError, fetch_app_details, fetch_patch_notes, fetch_reviews
from core.event_comments import fetch_event_comments, comments_to_dataframe

# Reviews older than this many days cannot be reliably fetched from Steam's
# recent-first API without paging through huge volumes of newer data.
_HISTORICAL_WARN_DAYS = 30


def scraper_node(state: PipelineState) -> dict:
    """
    Reads:  appid, update_date, pre_days, post_days
    Writes: game_name, pre_reviews, post_reviews, patch_notes,
            event_comments, current_step

    Data sources (in priority order):
      1. Local SQLite cache / live Steam review API.
      2. Steam partner events API → forum_topic_id → event comments.
      3. Patch note URLs as fallback for event comment GID resolution.
    """
    appid       = state["appid"]
    update_date = state["update_date"]
    pre_days    = state["pre_days"]
    post_days   = state["post_days"]

    try:
        # 1. Game metadata
        info      = fetch_app_details(appid)
        game_name = info["name"]

        # 2. Review windows
        now_utc      = datetime.now(timezone.utc)
        days_ago     = (now_utc - update_date).days
        errors: list[str] = []

        pre_start    = update_date - timedelta(days=pre_days)
        pre_end_excl = update_date - timedelta(seconds=1)
        post_end     = update_date + timedelta(days=post_days)

        pre_reviews  = fetch_reviews(appid, pre_start, pre_end_excl)
        post_reviews = fetch_reviews(appid, update_date, post_end)

        # 3. Historical-data guard
        if days_ago > _HISTORICAL_WARN_DAYS:
            if pre_reviews.empty and post_reviews.empty:
                errors.append(
                    f"[scraper] No reviews retrieved for either window. "
                    f"Update date is {days_ago} days ago. "
                    f"Run import_cache.py or use the UI cache initialiser."
                )
            else:
                missing = []
                if pre_reviews.empty:
                    missing.append("pre-update")
                if post_reviews.empty:
                    missing.append("post-update")
                if missing:
                    errors.append(
                        f"[scraper] Update date is {days_ago} days ago. "
                        f"No {' or '.join(missing)} reviews found. "
                        f"Results may be incomplete."
                    )

        # 4. Patch notes
        raw_notes   = fetch_patch_notes(appid, since_dt=pre_start)
        patch_notes = [PatchNote(**n) for n in raw_notes]

        # 5. Event comments — primary signal
        event_comments_df = _fetch_event_comments_via_partner_api(
            appid, update_date, patch_notes, errors,
        )

        output = ScraperOutput(
            appid=appid, game_name=game_name,
            n_pre_reviews=len(pre_reviews), n_post_reviews=len(post_reviews),
            n_patch_notes=len(patch_notes), n_event_comments=len(event_comments_df),
        )

        return {
            "game_name":      output.game_name,
            "pre_reviews":    pre_reviews,
            "post_reviews":   post_reviews,
            "patch_notes":    patch_notes,
            "event_comments": event_comments_df,
            "current_step":   "scraper_done",
            **({"errors": errors} if errors else {}),
        }

    except SteamAPIError as exc:
        return {
            "errors":       [f"[scraper] {exc}"],
            "current_step": "scraper_failed",
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_partner_events(appid: str, max_events: int = 100) -> list[dict]:
    """
    Fetch events from Steam's partner events API.
    Each event contains forum_topic_id — the GID for eventcomments endpoint.
    This works for BOTH old (Format A) and new (Format B) announcement pages.
    """
    events: list[dict] = []
    for offset in range(0, max_events, 20):
        try:
            r = requests.get(
                "https://store.steampowered.com/events/ajaxgetpartnereventspageable/",
                params={"appid": appid, "offset": offset, "count": 20, "l": "english"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            batch = r.json().get("events", [])
            if not batch:
                break
            events.extend(batch)
        except Exception:
            break
    return events


def _fetch_event_comments_via_partner_api(
    appid: str,
    update_date: datetime,
    patch_notes: list[PatchNote],
    errors: list[str],
):
    """
    Find the event closest to update_date, fetch its comments.

    Data flow:
      1. Check local cache first (instant).
      2. If cache miss: call partner events API → get forum_topic_id →
         scrape eventcomments → store in cache.

    The partner events API returns forum_topic_id — the correct GID for
    /app/{appid}/eventcomments/{ftid}, working for all announcement formats.
    """
    import pandas as pd
    from core.local_cache import (
        query_event_comments, store_event_comments, cached_event_topic_ids,
    )

    # Step 1: Get partner events (fast — just metadata, no comments)
    events = _fetch_partner_events(appid)
    if not events:
        errors.append("[scraper] Partner events API returned no events.")
        return _fallback_patch_note_comments(appid, patch_notes, update_date, errors)

    # Step 2: Sort events by proximity to update_date
    def _event_distance(e: dict) -> float:
        ts = e.get("rtime32_start_time", 0)
        if not ts:
            return float("inf")
        return abs(ts - update_date.timestamp())

    sorted_events = sorted(events, key=_event_distance)
    cached_ftids = set(cached_event_topic_ids(appid))

    # Step 3: Try each event — cache first, then live scrape
    for event in sorted_events[:20]:
        ftid = event.get("forum_topic_id", "")
        if not ftid:
            continue

        ts = event.get("rtime32_start_time", 0)
        event_date = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        event_name = event.get("event_name", "")
        event_gid  = event.get("gid", "")

        # Try cache first
        if ftid in cached_ftids:
            cached = query_event_comments(appid, ftid)
            if cached:
                df = comments_to_dataframe(
                    cached, fallback_timestamp=event_date or update_date,
                )
                if not df.empty:
                    errors.append(
                        f"[scraper] Event comments: {len(df)} from cache "
                        f"'{event_name}' (forum_topic_id={ftid})."
                    )
                    return df

        # Cache miss — scrape live
        raw = fetch_event_comments(appid, forum_topic_id=ftid)
        if raw:
            # Store in cache for next time
            date_str = event_date.strftime("%Y-%m-%d") if event_date else ""
            store_event_comments(
                appid, ftid, event_gid, event_name, date_str, raw,
            )

            df = comments_to_dataframe(
                raw, fallback_timestamp=event_date or update_date,
            )
            if not df.empty:
                delta_str = ""
                if event_date:
                    delta_days = (event_date - update_date).days
                    delta_str = f", {delta_days:+d}d from update"
                errors.append(
                    f"[scraper] Event comments: {len(df)} from "
                    f"'{event_name}' (forum_topic_id={ftid}{delta_str}, live scrape -> cached)."
                )
                return df

    errors.append(
        f"[scraper] Checked {min(len(sorted_events), 20)} partner events "
        f"-- no comments found."
    )
    return _fallback_patch_note_comments(appid, patch_notes, update_date, errors)


def _fallback_patch_note_comments(
    appid: str,
    patch_notes: list[PatchNote],
    update_date: datetime,
    errors: list[str],
):
    """
    Legacy fallback: try patch note URLs/GIDs for event comments.
    Used when partner events API is unavailable.
    """
    import pandas as pd

    if not patch_notes:
        return pd.DataFrame()

    sorted_notes = sorted(
        patch_notes,
        key=lambda n: abs((n.published_at - update_date).total_seconds()),
    )

    for note in sorted_notes:
        try:
            raw = fetch_event_comments(appid, news_url=note.url, news_gid=note.gid)
            if raw:
                df = comments_to_dataframe(raw, fallback_timestamp=note.published_at)
                if not df.empty:
                    return df
        except Exception:
            continue

    return pd.DataFrame()
