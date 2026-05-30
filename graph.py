# graph.py
# Assembles the LangGraph pipeline for Steam Update Impact Analysis.
#
# Graph flow:
#
#   scraper → cleaning → analysis ──(gray-zone reviews)──► llm_sentiment ─┐
#                                └────────────────────────────────────────►┘
#                                                                          │
#                                                                    recommendation
#                                                                          │
#                                                                         END
#
# Why no llm_review node?
# -----------------------
# Steam enforces a minimum review length (~20 chars), so the cleaning rules
# (short text, repeated chars, high special-char ratio) have a 0% trigger rate
# on real Steam review data — verified across 5 update windows on CS2 (71k reviews).
# The conditional branch was dead code and has been removed.

from langgraph.graph import END, StateGraph

from agents.analysis_agent import analysis_node, should_run_llm_sentiment
from agents.cleaning_agent import cleaning_node
from agents.llm_sentiment_agent import llm_sentiment_node
from agents.recommendation_agent import recommendation_node
from agents.scraper_agent import scraper_node
from agents.state import PipelineState


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("scraper",        scraper_node)
    g.add_node("cleaning",       cleaning_node)
    g.add_node("analysis",       analysis_node)
    g.add_node("llm_sentiment",  llm_sentiment_node)
    g.add_node("recommendation", recommendation_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("scraper")

    # ── Edges ─────────────────────────────────────────────────────────────────
    g.add_edge("scraper", "cleaning")
    g.add_edge("cleaning", "analysis")

    # Analysis → LLM sentiment re-score  OR  straight to recommendation
    g.add_conditional_edges(
        "analysis",
        should_run_llm_sentiment,
        {"llm_sentiment": "llm_sentiment", "recommendation": "recommendation"},
    )
    g.add_edge("llm_sentiment", "recommendation")

    g.add_edge("recommendation", END)

    return g


# Compiled graph — import this in the UI and tests
app = build_graph().compile()
