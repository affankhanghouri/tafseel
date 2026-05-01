import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from src.state import MyState
from src.prompts import (
    decide_retrieval_prompt,
    direct_generation_prompt,
    rag_generation_prompt,
    is_relevant_prompt,
    issup_prompt,
    revise_prompt,
    isuse_prompt,
    rewrite_for_retrieval_prompt,
    VOICE_DIRECT_PROMPTS,
    VOICE_RAG_PROMPTS,
    LANGUAGE_INSTRUCTIONS,
)
from src.routing import llm, voice_llm, decision_llm, relevance_llm, issup_llm, isuse_llm, rewrite_llm
from src.ingestion import get_retriever

logger = logging.getLogger(__name__)

HISTORY_WINDOW = 5   # last 5 turns (user+assistant pairs) passed to generation nodes

# Supported languages — fallback to urdu if unknown
SUPPORTED_LANGUAGES = {"urdu", "sindhi", "balochi", "english"}


def _resolve_language(state: MyState) -> str:
    """Return a validated, lowercase language string."""
    lang = state.get("language", "urdu").lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        logger.warning(f"[language] Unknown language '{lang}' — defaulting to 'urdu'")
        return "urdu"
    return lang


def _build_history_messages(state: MyState) -> list:
    """Convert state history → LangChain message objects for {history} placeholder."""
    history = state.get("history", [])
    recent = history[-(HISTORY_WINDOW * 2):]
    messages = []
    for msg in recent:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ─────────────────────────────────────────────
# Router node — language-agnostic
# ─────────────────────────────────────────────
def decide_retrieval(state: MyState):
    try:
        decision = decision_llm.invoke(
            decide_retrieval_prompt.format_messages(question=state["question"])
        )
        return {"need_retrieval": decision.need_retrieval}
    except Exception as e:
        logger.error(f"[decide_retrieval] Structured output failed: {e} — defaulting to True")
        return {"need_retrieval": True}


# ─────────────────────────────────────────────
# Direct generation (no retrieval)
# Voice: language-specific voice prompt
# Text: unified prompt with language instruction injected
# ─────────────────────────────────────────────
def generate_direct(state: MyState):
    is_voice = state.get("mode") == "voice"
    language = _resolve_language(state)
    history_messages = _build_history_messages(state)

    try:
        if is_voice:
            prompt = VOICE_DIRECT_PROMPTS[language]
            ans = voice_llm.invoke(
                prompt.format_messages(
                    question=state["question"],
                    history=history_messages,
                )
            )
        else:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["urdu"])
            ans = llm.invoke(
                direct_generation_prompt.format_messages(
                    question=state["question"],
                    history=history_messages,
                    language_instruction=lang_instruction,
                )
            )
        return {"answer": ans.content}
    except Exception as e:
        logger.error(f"[generate_direct] LLM call failed: {e}")
        return {"answer": _fallback_message(language, is_voice)}


# ─────────────────────────────────────────────
# Retrieval node — Qdrant failure returns empty docs (no crash)
# ─────────────────────────────────────────────
def retrieve(state: MyState):
    query = state.get("retrieval_query") or state["question"]
    try:
        retrieved_docs = get_retriever().invoke(query)
        return {"docs": retrieved_docs}
    except Exception as e:
        logger.error(f"[retrieve] Qdrant retrieval failed: {e} — returning empty docs")
        return {"docs": []}


# ─────────────────────────────────────────────
# Relevance grader — PARALLEL via ThreadPoolExecutor
# ─────────────────────────────────────────────
def _grade_single_doc(doc: Document, question: str):
    """Grade one document. Returns (doc, is_relevant)."""
    try:
        decision = relevance_llm.invoke(
            is_relevant_prompt.format_messages(question=question, document=doc.page_content)
        )
        return doc, decision.is_relevant
    except Exception as e:
        logger.warning(f"[is_relevant] Grading failed — excluding doc (accuracy > recall): {e}")
        return doc, False   # fail safe: exclude rather than risk a hallucinated answer


def is_relevant(state: MyState):
    docs = state.get("docs", [])
    if not docs:
        return {"relevant_docs": []}

    question = state["question"]
    relevant_docs: List[Document] = []

    with ThreadPoolExecutor(max_workers=len(docs)) as executor:
        futures = {executor.submit(_grade_single_doc, doc, question): doc for doc in docs}
        for future in as_completed(futures):
            doc, is_rel = future.result()
            if is_rel:
                relevant_docs.append(doc)

    # Restore original ordering
    order = {id(d): i for i, d in enumerate(docs)}
    relevant_docs.sort(key=lambda d: order.get(id(d), 999))

    logger.info(f"[is_relevant] {len(relevant_docs)}/{len(docs)} docs kept")
    return {"relevant_docs": relevant_docs}


# ─────────────────────────────────────────────
# RAG generation — language-aware
# ─────────────────────────────────────────────
def generate_from_context(state: MyState):
    context = "\n\n---\n\n".join(
        d.page_content for d in state.get("relevant_docs", [])
    ).strip()

    if not context:
        return {"answer": "No answer found.", "context": ""}

    is_voice = state.get("mode") == "voice"
    language = _resolve_language(state)
    history_messages = _build_history_messages(state)

    try:
        if is_voice:
            prompt = VOICE_RAG_PROMPTS[language]
            out = voice_llm.invoke(
                prompt.format_messages(
                    question=state["question"],
                    context=context,
                    history=history_messages,
                )
            )
        else:
            lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["urdu"])
            out = llm.invoke(
                rag_generation_prompt.format_messages(
                    question=state["question"],
                    context=context,
                    history=history_messages,
                    language_instruction=lang_instruction,
                )
            )
        return {"answer": out.content, "context": context}
    except Exception as e:
        logger.error(f"[generate_from_context] LLM call failed: {e}")
        return {
            "answer": _fallback_message(language, is_voice),
            "context": context,
        }


# ─────────────────────────────────────────────
# Terminal: no relevant docs found
# ─────────────────────────────────────────────
def no_answer_found(state: MyState):
    language = _resolve_language(state)
    is_voice = state.get("mode") == "voice"
    return {"answer": _no_answer_message(language, is_voice), "context": ""}


# ─────────────────────────────────────────────
# IsSUP hallucination check
# ─────────────────────────────────────────────
def is_sup(state: MyState):
    try:
        decision = issup_llm.invoke(
            issup_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        return {"issup": decision.issup, "evidence": decision.evidence}
    except Exception as e:
        logger.error(f"[is_sup] Structured output failed: {e} — defaulting to fully_supported")
        return {"issup": "fully_supported", "evidence": []}


# ─────────────────────────────────────────────
# Revise answer — preserves language automatically (prompt instructs the LLM)
# ─────────────────────────────────────────────
def revise_answer(state: MyState):
    is_voice = state.get("mode") == "voice"
    language = _resolve_language(state)
    model = voice_llm if is_voice else llm
    try:
        out = model.invoke(
            revise_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
                context=state.get("context", ""),
            )
        )
        return {"answer": out.content, "retries": state.get("retries", 0) + 1}
    except Exception as e:
        logger.error(f"[revise_answer] LLM call failed: {e} — keeping current answer")
        return {
            "answer": state.get("answer", _fallback_message(language, is_voice)),
            "retries": state.get("retries", 0) + 1,
        }


# ─────────────────────────────────────────────
# IsUSE usefulness check
# ─────────────────────────────────────────────
def is_use(state: MyState):
    try:
        decision = isuse_llm.invoke(
            isuse_prompt.format_messages(
                question=state["question"],
                answer=state.get("answer", ""),
            )
        )
        return {"is_useful": decision.is_useful}
    except Exception as e:
        logger.error(f"[is_use] Structured output failed: {e} — defaulting to True")
        return {"is_useful": True}


# ─────────────────────────────────────────────
# Rewrite question for better retrieval
# Always rewrites to English for consistent vector search
# ─────────────────────────────────────────────
def rewrite_question(state: MyState):
    try:
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
            "docs": [],
            "relevant_docs": [],
            "context": "",
        }
    except Exception as e:
        logger.error(f"[rewrite_question] Structured output failed: {e} — using original question")
        return {
            "retrieval_query": state["question"],
            "rewrite_tries": state.get("rewrite_tries", 0) + 1,
            "docs": [],
            "relevant_docs": [],
            "context": "",
        }


# ─────────────────────────────────────────────
# Fallback messages — per language, per mode
# ─────────────────────────────────────────────
def _fallback_message(language: str, is_voice: bool) -> str:
    """Generic error fallback in the correct language."""
    messages = {
        "urdu":    "معذرت، ایک مسئلہ پیش آیا۔ براہ کرم دوبارہ کوشش کریں یا NADRA ہیلپ لائن 1 7 7 7 پر کال کریں۔",
        "sindhi":  "معاف ڪجو، هڪ مسئلو پيش آيو۔ مهرباني ڪري ٻيهر ڪوشش ڪريو يا NADRA helpline 1 7 7 7 تي call ڪريو۔",
        "balochi": "معاف کن، یک مشکل پیش آمد۔ دوباره امتحان بکن یا NADRA helpline 1 7 7 7 را زنگ بزن۔",
        "english": "I'm sorry, I encountered an issue. Please try again or call the NADRA helpline at 1-7-7-7.",
    }
    return messages.get(language, messages["urdu"])


def _no_answer_message(language: str, is_voice: bool) -> str:
    """No-answer message in the correct language."""
    messages = {
        "urdu":    "معذرت، اس سوال کا جواب میسر نہیں ہے۔ مزید مدد کے لیے NADRA ہیلپ لائن 1 7 7 7 پر کال کریں یا complaints.nadra.gov.pk پر جائیں۔",
        "sindhi":  "معاف ڪجو، هن سوال جو جواب موجود ناهي۔ وڌيڪ مدد لاءِ NADRA helpline 1 7 7 7 تي call ڪريو يا complaints.nadra.gov.pk تي وڃو۔",
        "balochi": "معاف کن، این سوال جواب موجود نیست۔ بیشتر کمک لئی NADRA helpline 1 7 7 7 را زنگ بزن یا complaints.nadra.gov.pk را ببین۔",
        "english": "I'm sorry, I couldn't find an answer to your question. For further help, please call the NADRA helpline at 1-7-7-7 or visit complaints.nadra.gov.pk.",
    }
    return messages.get(language, messages["urdu"])