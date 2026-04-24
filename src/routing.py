import os
from typing import Literal

# BUG FIX #1: was "langchain_graphraphroq" (typo) — correct package is "langchain_groq"
from langchain_groq import ChatGroq

from src.models import (
    RetrievalDecision,
    RelevanceDecision,
    IsSUPDecision,
    IsUSEDecision,
    RewriteDecision,
)
from src.state import MyState
from src.config import MAX_RETRIES, MAX_REWRITE_TRIES

# ─────────────────────────────────────────────
# LLM + structured output variants
# ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

decision_llm  = llm.with_structured_output(RetrievalDecision)
relevance_llm = llm.with_structured_output(RelevanceDecision)
issup_llm     = llm.with_structured_output(IsSUPDecision)
isuse_llm     = llm.with_structured_output(IsUSEDecision)
rewrite_llm   = llm.with_structured_output(RewriteDecision)

# ─────────────────────────────────────────────
# Conditional edge: after decide_retrieval node
# BUG FIX #3: was named "route_decider" in node cell but "route_after_decide"
# in graph cell — now consistent everywhere
# ─────────────────────────────────────────────
def route_after_decide(state: MyState) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


# ─────────────────────────────────────────────
# Conditional edge: after is_relevant node
# ─────────────────────────────────────────────
def route_after_relevance(
    state: MyState,
) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


# ─────────────────────────────────────────────
# Conditional edge: after is_sup node
# ─────────────────────────────────────────────
def route_after_issup(
    state: MyState,
) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"   # max retries hit → pass to IsUSE anyway
    return "revise_answer"


# ─────────────────────────────────────────────
# Conditional edge: after is_use node
# BUG FIX #4: this function was referenced in the graph but never defined
# ─────────────────────────────────────────────
def route_after_isuse(
    state: MyState,
) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("is_useful"):
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"