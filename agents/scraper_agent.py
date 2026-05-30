# agents/scraper_agent.py
# LangGraph node: fetch Steam reviews + patch notes + event comments for an
# update event.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.state import PatchNote, PipelineState, ScraperOutput
from core.scraper import SteamAPIError, fetch_app_details, fetch_patch_notes, fetch_reviews
from core.event_comments import fetch_event_comments, comments_to_dataframe
from config import EVENT_COMMENT_WEIGHT

# Reviews older than this many days cannot be reliably fetched from Steam's
# recent-first API without paging through huge volumes of newer data.
# Popular games (CS2, Dota 2) may effectively cap out even sooner.
_HISTORICAL_WARN_DAYS = 30


def scraper_node(state: PipelineState) -> dict:
    """
    Reads:  appid, update_date, pre_days, post_days
    Writes: game_name, pre_reviews, post_reviews, patch_notes,
            event_comments, current_step

    Data sources (in priority order):
      1. Local SQLite cache — for historical windows already scraped.
      2. Live Steam review API — for recent windows (< ~30 days).
      3. Steam event comments — fetched for the patch note closest to
         update_date; treated as high-signal post-update data.

    Window boundaries (no overlap / no leakage):
      pre  : [update_date - pre_days,  update_date)   exclusive right edge
      post : [update_date,             update_date + post_days]  inclusive
    Reviews at exactly update_date are counted as post-update only.
    """
    appid       = state["appid"]
    update_date = state["update_date"]
    pre_days    = state["pre_days"]
    post_days   = state["post_days"]

    try:
        # 1. Game metadata
        info      = fetch_app_details(appid)
        game_name = info["name"]

        # 2. Review windows — 1-second gap prevents boundary overlap
        #    pre  : [update_date − pre_days,  update_date)   exclusive right edge
        #    post : [update_date,             update_date + post_days]
        now_utc      = datetime.now(timezone.utc)
        days_ago     = (now_utc - update_date).days
        errors: list[str] = []

        pre_start    = update_date - timedelta(days=pre_days)
        pre_end_excl = update_date - timedelta(seconds=1)
        post_end     = update_date + timedelta(days=post_days)

        pre_reviews  = fetch_reviews(appid, pre_start, pre_end_excl)
        post_reviews = fetch_reviews(appid, update_date, post_end)

        # 3. Historical-data guard — only warn when data is actually missing
        if days_ago > _HISTORICAL_WARN_DAYS:
            if pre_reviews.empty and post_reviews.empty:
                errors.append(
                    f"[scraper] ❌  No reviews retrieved for either window. "
                    f"Update date is {days_ago} days ago and the local cache has no data "
                    f"for this window.  Run import_cache.py to load historical data, "
                    f"or try an update from the last {_HISTORICAL_WARN_DAYS} days."
                )
            else:
                missing = []
                if pre_reviews.empty:
                    missing.append("pre-update")
                if post_reviews.empty:
                    missing.append("post-update")
                if missing:
                    errors.append(
                        f"[scraper] ⚠️  Update date is {days_ago} days ago. "
                        f"No {' or '.join(missing)} reviews found in local cache for this window. "
                        f"Results may be incomplete."
                    )

        # 4. Patch notes published within the pre_days look-back
        raw_notes   = fetch_patch_notes(appid, since_dt=pre_start)
        patch_notes = [PatchNote(**n) for n in raw_notes]

        # 5. Event comments — fetch from the patch note closest to update_date
        #    Event comments are anchored to the specific announcement and carry
        #    higher signal than general reviews for that update.
        event_comments_df = _fetch_closest_event_comments(
            appid, patch_notes, update_date, errors
        )

        # Validate output shape
        output = ScraperOutput(
            appid=             appid,
            game_name=         game_name,
            n_pre_reviews=     len(pre_reviews),
            n_post_reviews=    len(post_reviews),
            n_patch_notes=     len(patch_notes),
            n_event_comments=  len(event_comments_df),
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

def _fetch_closest_event_comments(
    appid: str,
    patch_notes: list[PatchNote],
    update_date: datetime,
    errors: list[str],
):
    """
    Find the patch note closest to (and not after) update_date + 1 day, then
    fetch its event comments.  Returns an empty DataFrame on any failure.
    """
    import pandas as pd

    if not patch_notes:
        return pd.DataFrame()

    # Pick the patch note whose published_at is closest to update_date
    # (within a ±2-day window to allow for announcement timing)
    _2d = timedelta(days=2)
    candidates = [
        n for n in patch_notes
        if abs((n.published_at - update_date).total_seconds()) <= _2d.total_seconds()
    ]
    if not candidates:
        # Fallback: just use the most recent patch note
        candidates = sorted(patch_notes, key=lambda n: n.published_at, reverse=True)

    note = min(candidates, key=lambda n: abs((n.published_at - update_date).total_seconds()))

    try:
        raw = fetch_event_comments(appid, news_url=note.url, news_gid=note.gid)
        # Pass the patch note's published_at as fallback timestamp so comments
        # without a parseable timestamp are still treated as post-update data.
        df  = comments_to_dataframe(raw, fallback_timestamp=note.published_at,
                                    weight=EVENT_COMMENT_WEIGHT)
        if not df.empty:
            return df
        return pd.DataFrame()
    except Exception as exc:
        errors.append(f"[scraper] ⚠️  Event comments unavailable: {exc}")
        return pd.DataFrame()
