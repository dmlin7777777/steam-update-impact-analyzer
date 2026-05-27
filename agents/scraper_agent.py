# agents/scraper_agent.py
# LangGraph node: fetch Steam reviews + patch notes for an update event.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agents.state import PatchNote, PipelineState, ScraperOutput
from core.scraper import SteamAPIError, fetch_app_details, fetch_patch_notes, fetch_reviews

# Reviews older than this many days cannot be reliably fetched from Steam's
# recent-first API without paging through huge volumes of newer data.
# Popular games (CS2, Dota 2) may effectively cap out even sooner.
_HISTORICAL_WARN_DAYS = 30


def scraper_node(state: PipelineState) -> dict:
    """
    Reads:  appid, update_date, pre_days, post_days
    Writes: game_name, pre_reviews, post_reviews, patch_notes, current_step

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

        # 3. Historical-data guard — only warn when data is actually missing/sparse
        #    (if cache served the data, no warning is needed)
        if days_ago > _HISTORICAL_WARN_DAYS:
            if pre_reviews.empty and post_reviews.empty:
                errors.append(
                    f"[scraper] ❌  No reviews retrieved for either window. "
                    f"Update date is {days_ago} days ago and the local cache has no data "
                    f"for this window.  Run scripts/import_cache.py to load historical data, "
                    f"or try an update from the last {_HISTORICAL_WARN_DAYS} days."
                )
            else:
                # Partial data: warn only if either window is empty
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

        # Validate output shape
        output = ScraperOutput(
            appid=          appid,
            game_name=      game_name,
            n_pre_reviews=  len(pre_reviews),
            n_post_reviews= len(post_reviews),
            n_patch_notes=  len(patch_notes),
        )

        return {
            "game_name":    output.game_name,
            "pre_reviews":  pre_reviews,
            "post_reviews": post_reviews,
            "patch_notes":  patch_notes,
            "current_step": "scraper_done",
            **({"errors": errors} if errors else {}),
        }

    except SteamAPIError as exc:
        return {
            "errors":       [f"[scraper] {exc}"],
            "current_step": "scraper_failed",
        }
