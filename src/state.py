from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class MyState(TypedDict):
    # Core input
    question: str
    mode: str                  # "text" or "voice"

    # Routing
    need_retrieval: bool

    # Retrieval
    docs: List[Document]
    relevant_docs: List[Document]
    retrieval_query: str       # rewritten query used by retrieve node
    rewrite_tries: int         # number of rewrite attempts made

    # Generation
    context: str
    answer: str

    # IsSUP grading loop
    issup: str                 # "fully_supported" | "partially_supported" | "no_support"
    evidence: List[str]
    retries: int               # number of revise-answer attempts made

    # IsUSE usefulness check
    is_useful: bool