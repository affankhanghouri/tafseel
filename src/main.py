from graph import app


if __name__ == "__main__":
    question = input("Ask your question: ")

    # BUG FIX #10: original invocation was missing all new state fields
    # which would cause KeyError at runtime
    result = app.invoke({
        "question":       question,
        "need_retrieval": False,
        "docs":           [],
        "relevant_docs":  [],
        "context":        "",
        "answer":         "",
        "retrieval_query": "",
        "retries":        0,
        "rewrite_tries":  0,
        "issup":          "",
        "evidence":       [],
        "is_useful":      False,
    })

    print("\nFinal Answer:\n")
    print(result["answer"])
    print("other information")
    print()
    print(result)