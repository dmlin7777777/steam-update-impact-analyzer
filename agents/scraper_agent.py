# agents/scraper_agent.py
# LangGraph node: fetch Steam reviews + patch notes for an update event.

from __future__ import annotations

from datetime import timedelta

from agents.state import PatchNote, PipelineState, ScraperOutput
from core.scraper import SteamAPIError, fetch_app_details, fetch_patch_notes, fetch_reviews


def scraper_node(state: PipelineState) -> dict:
    """
    Reads:  appid, update_date, pre_days, post_days
    Writes: game_name, pre_reviews, post_reviews, patch_notes, current_step
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
        pre_start  = update_date - timedelta(days=pre_days)
        post_end   = update_date + timedelta(days=post_days)

        pre_reviews  = fetch_reviews(appid, pre_start,   update_date)
        post_reviews = fetch_reviews(appid, update_date, post_end)

        # 3. Patch notes published within the pre_days look-back
        raw_notes  = fetch_patch_notes(appid, since_dt=pre_start)
        patch_notes = [PatchNote(**n) for n in raw_notes]

        # Validate output shape
        output = ScraperOutput(
            appid=         appid,
            game_name=     game_name,
            n_pre_reviews= len(pre_reviews),
            n_post_reviews=len(post_reviews),
            n_patch_notes= len(patch_notes),
        )

        return {
            "game_name":    output.game_name,
            "pre_reviews":  pre_reviews,
            "post_reviews": post_reviews,
            "patch_notes":  patch_notes,
            "current_step": "scraper_done",
        }

    except SteamAPIError as exc:
        return {
            "errors":       [f"[scraper] {exc}"],
            "current_step": "scraper_failed",
        }
