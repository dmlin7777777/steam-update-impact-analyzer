# graph.py
# Assembles the LangGraph pipeline for Steam Update Impact Analysis.
#
# Graph flow:
#
#   scraper
#     └─► cleaning ──(flagged > threshold)──► llm_review ─┐
#                  └──────────────────────────────────────►┘
#                                                          │
#                                                       analysis
#                                                          │
#                           ┌──(gray-zone reviews)─────────┘
#                        llm_sentiment                     │
#                           └──────────────────────────────►┘
#                                                          │
#                                                    recommendation
#                                                          │
#                                                         END

from langgraph.graph import END, StateGraph

from agents.analysis_agent import analysis_node, should_run_llm_sentiment
from agents.cleaning_agent import cleaning_node, should_run_llm_review
from agents.llm_review_agent import llm_review_node
from agents.llm_sentiment_agent import llm_sentiment_node
from agents.recommendation_agent import recommendation_node
from agents.scraper_agent import scraper_node
from agents.state import PipelineState


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("scraper",        scraper_node)
    g.add_node("cleaning",       cleaning_node)
    g.add_node("llm_review",     llm_review_node)
    g.add_node("analysis",       analysis_node)
    g.add_node("llm_sentiment",  llm_sentiment_node)
    g.add_node("recommendation", recommendation_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("scraper")

    # ── Edges ─────────────────────────────────────────────────────────────────
    g.add_edge("scraper", "cleaning")

    # Cleaning → LLM review  OR  straight to analysis
    g.add_conditional_edges(
        "cleaning",
        should_run_llm_review,
        {"llm_review": "llm_review", "analysis": "analysis"},
    )
    g.add_edge("llm_review", "analysis")

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
