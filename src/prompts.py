from langchain_core.prompts import ChatPromptTemplate

# ─────────────────────────────────────────────
# Router: decide whether retrieval is needed
# ─────────────────────────────────────────────
decide_retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert router. Determine if the question is about NADRA documents/rules "
        "or general knowledge.\n"
        "Return JSON: {{'need_retrieval': boolean}}\n\n"
        "Guidelines:\n"
        "- need_retrieval=True: Specific facts about NADRA, CNIC, Juvenile cards, or POC.\n"
        "- need_retrieval=False: General knowledge (e.g., 'What is Machine Learning?'), "
        "greetings, or math.\n"
        "- If unsure, choose True.",
    ),
    ("human", "Question: {question}"),
])

# ─────────────────────────────────────────────
# Generation: direct LLM answer (no retrieval)
# ─────────────────────────────────────────────
direct_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using only your general knowledge.\n"
        "If you are unsure, say 'I don't know based on my general knowledge.'",
    ),
    ("human", "{question}"),
])

# ─────────────────────────────────────────────
# Grader: document relevance check
# ─────────────────────────────────────────────
is_relevant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are judging document relevance at a TOPIC level.\n"
        "Return JSON matching the schema.\n\n"
        "A document is relevant if it discusses the same entity or topic area as the question.\n"
        "It does NOT need to contain the exact answer.\n\n"
        "Examples:\n"
        "- NADRA policy docs are relevant to questions about CNIC, POC, Juvenile card, or renewal.\n"
        "- Office location documents are relevant to questions about NADRA offices or addresses.\n\n"
        "Do NOT decide whether the document fully answers the question.\n"
        "That will be checked later by IsSUP.\n"
        "When unsure, return is_relevant=true.",
    ),
    ("human", "Question:\n{question}\n\nDocument:\n{document}"),
])

# ─────────────────────────────────────────────
# Generation: RAG answer from retrieved context
# ─────────────────────────────────────────────
rag_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a NADRA Intelligent Assistant (NIA).\n\n"
        "You will receive a CONTEXT block from internal NADRA documents.\n"
        "Answer the question based ONLY on the context.\n"
        "Do not mention that you are receiving a context in your answer.",
    ),
    ("human", "Question:\n{question}\n\nContext:\n{context}"),
])

# ─────────────────────────────────────────────
# Grader: IsSUP — is the answer supported by context?
# ─────────────────────────────────────────────
issup_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
        "Return JSON with keys: issup, evidence.\n"
        "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
        "How to decide issup:\n"
        "- fully_supported:\n"
        "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT "
        "introduce any qualitative/interpretive words not present in CONTEXT.\n\n"
        "- partially_supported:\n"
        "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, "
        "or qualitative phrasing not explicitly stated in CONTEXT.\n\n"
        "- no_support:\n"
        "  The key claims are not supported by CONTEXT.\n\n"
        "Rules:\n"
        "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, "
        "choose partially_supported.\n"
        "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
        "- Evidence: include up to 3 short direct quotes from CONTEXT that support the "
        "supported parts.\n"
        "- Do not use outside knowledge.",
    ),
    (
        "human",
        "Question:\n{question}\n\n"
        "Answer:\n{answer}\n\n"
        "Context:\n{context}\n",
    ),
])

# ─────────────────────────────────────────────
# Reviser: rewrite answer to be context-only
# ─────────────────────────────────────────────
revise_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a STRICT reviser.\n\n"
        "FORMAT (quote-only answer):\n"
        "- <direct quote from the CONTEXT>\n"
        "- <direct quote from the CONTEXT>\n\n"
        "Rules:\n"
        "- Use ONLY the CONTEXT.\n"
        "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
        "- Do NOT explain anything.\n"
        "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n",
    ),
    (
        "human",
        "Question:\n{question}\n\n"
        "Current Answer:\n{answer}\n\n"
        "CONTEXT:\n{context}",
    ),
])

# ─────────────────────────────────────────────
# Grader: IsUSE — is the answer useful to the user?
# ─────────────────────────────────────────────
isuse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are checking whether the ANSWER actually addresses the user's QUESTION.\n"
        "Return JSON with key: is_useful (boolean).\n"
        "is_useful=True: The answer directly and helpfully answers the question.\n"
        "is_useful=False: The answer is off-topic, says 'not found', or is evasive.",
    ),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
])

# ─────────────────────────────────────────────
# Rewriter: rewrite query for better retrieval
# ─────────────────────────────────────────────
rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's QUESTION into a query optimized for vector retrieval over "
        "NADRA internal documents.\n\n"
        "Rules:\n"
        "- Keep it short (6–16 words).\n"
        "- Preserve key entities (e.g., CNIC, NADRA, POC, Juvenile card).\n"
        "- Add 2–5 high-signal keywords that likely appear in NADRA policy or office docs.\n"
        "- Remove filler words.\n"
        "- Do NOT answer the question.\n"
        "- Output JSON with key: retrieval_query\n\n"
        "Examples:\n"
        "Q: 'How do I renew my CNIC?'\n"
        "-> {{'retrieval_query': 'CNIC renewal process requirements documents fee NADRA'}}\n\n"
        "Q: 'What is a POC card?'\n"
        "-> {{'retrieval_query': 'POC Pakistan Origin Card eligibility application process'}}",
    ),
    (
        "human",
        "QUESTION:\n{question}\n\n"
        "Previous retrieval query:\n{retrieval_query}\n\n"
        "Answer (if any):\n{answer}",
    ),
])