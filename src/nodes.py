from typing import List

from langchain_core.documents import Document

from src.state import MyState
from src.prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    is_relevant_prompt,
    rag_generation_prompt,
    issup_prompt,
    revise_prompt,
    isuse_prompt,
    rewrite_for_retrieval_prompt,
)
from src.routing import llm, decision_llm, relevance_llm, issup_llm, isuse_llm, rewrite_llm
from src.ingestion import retriever


# ─────────────────────────────────────────────
# Router node: decide whether retrieval is needed
# ─────────────────────────────────────────────
def decide_retrieval(state: MyState):
    decision = decision_llm.invoke(
        decide_retrieval_prompt.format_messages(question=state["question"])
    )
    return {"need_retrieval": decision.need_retrieval}


# ─────────────────────────────────────────────
# Generation node: direct LLM answer (no retrieval path)
# ─────────────────────────────────────────────
def generate_direct(state: MyState):
    ans = llm.invoke(
        direct_generation_prompt.format_messages(question=state["question"])
    )
    return {"answer": ans.content}


# ─────────────────────────────────────────────
# Retrieval node
# BUG FIX #7: original always used state["question"]; after a rewrite the
# node must use the rewritten query when available
# ─────────────────────────────────────────────
def retrieve(state: MyState):
    query = state.get("retrieval_query") or state["question"]
    retrieved_docs = retriever.invoke(query)
    return {"docs": retrieved_docs}


# ─────────────────────────────────────────────
# Grader node: filter docs by relevance
# ─────────────────────────────────────────────
def is_relevant(state: MyState):
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


# ─────────────────────────────────────────────
# Generation node: RAG answer from relevant docs
# ─────────────────────────────────────────────
def generate_from_context(state: MyState):
    context = "\n\n---\n\n".join(
        d.page_content for d in state.get("relevant_docs", [])
    ).strip()

    if not context:
        return {"answer": "No answer found.", "context": ""}

    out = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context,
        )
    )
    return {"answer": out.content, "context": context}


# ─────────────────────────────────────────────
# Terminal node: no relevant docs found
# ─────────────────────────────────────────────
def no_answer_found(state: MyState):
    return {"answer": "No answer found.", "context": ""}


# ─────────────────────────────────────────────
# Grader node: IsSUP — is answer supported by context?
# ─────────────────────────────────────────────
def is_sup(state: MyState):
    decision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


# ─────────────────────────────────────────────
# Revision node: rewrite answer to be context-only
# ─────────────────────────────────────────────
def revise_answer(state: MyState):
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }


# ─────────────────────────────────────────────
# Grader node: IsUSE — is the answer useful?
# BUG FIX #4: this node was added to the graph but never defined
# ─────────────────────────────────────────────
def is_use(state: MyState):
    decision = isuse_llm.invoke(
        isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"is_useful": decision.is_useful}


# ─────────────────────────────────────────────
# Rewriter node: rewrite question for better retrieval
# ─────────────────────────────────────────────
def rewrite_question(state: MyState):
    decision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        # reset so next retrieval pass is clean
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }