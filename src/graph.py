from langgraph.graph import StateGraph, START, END

from src.state import MyState
from src.nodes import (
    decide_retrieval,
    generate_direct,
    retrieve,
    is_relevant,
    generate_from_context,
    no_answer_found,
    is_sup,
    revise_answer,
    is_use,
    rewrite_question,
)
from src.routing import (
    route_after_decide,
    route_after_relevance,
    route_after_issup,
    route_after_isuse,
)


def build_graph() -> StateGraph:
    graph = StateGraph(MyState)

    # ── Nodes ──────────────────────────────────────────
    graph.add_node("decide_retrieval",    decide_retrieval)
    graph.add_node("generate_direct",     generate_direct)
    graph.add_node("retrieve",            retrieve)
    graph.add_node("is_relevant",         is_relevant)
    graph.add_node("generate_from_context", generate_from_context)
    graph.add_node("no_answer_found",     no_answer_found)
    graph.add_node("is_sup",              is_sup)
    graph.add_node("revise_answer",       revise_answer)
    graph.add_node("is_use",              is_use)
    graph.add_node("rewrite_question",    rewrite_question)

    # ── Edges ──────────────────────────────────────────

    # Entry point
    graph.add_edge(START, "decide_retrieval")

    # Router: NADRA query → retrieve | general query → direct LLM
    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )

    # Direct path ends immediately
    graph.add_edge("generate_direct", END)

    # Retrieval path: retrieve → grade relevance
    graph.add_edge("retrieve", "is_relevant")

    # Relevance gate: relevant docs → generate | no docs → no_answer_found
    graph.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )

    graph.add_edge("no_answer_found", END)

    # Generation → IsSUP hallucination check
    graph.add_edge("generate_from_context", "is_sup")

    # IsSUP loop: fully supported → IsUSE | not supported → revise (up to MAX_RETRIES)
    graph.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {
            "accept_answer": "is_use",       # "accept_answer" is a routing label, not a node
            "revise_answer": "revise_answer",
        },
    )

    graph.add_edge("revise_answer", "is_sup")  # loop back

    # IsUSE: useful → END | not useful → rewrite | max rewrites → no_answer_found
    graph.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "END": END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )

    # Rewrite → retrieve again (new retrieval_query will be used)
    graph.add_edge("rewrite_question", "retrieve")

    return graph.compile()


app = build_graph()