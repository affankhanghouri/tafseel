from src.graph import app

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    question = input("Ask your question: ")
    language = input("Language (urdu/sindhi/balochi/english) [urdu]: ").strip().lower() or "urdu"

    result = app.invoke({
        "question":         question,
        "mode":             "text",
        "language":         language,
        "conversation_id":  None,
        "history":          [],
        "need_retrieval":   False,
        "docs":             [],
        "relevant_docs":    [],
        "context":          "",
        "answer":           "",
        "retrieval_query":  "",
        "retries":          0,
        "rewrite_tries":    0,
        "issup":            "",
        "evidence":         [],
        "is_useful":        False,
    })

    print("\nFinal Answer:\n")
    print(result["answer"])