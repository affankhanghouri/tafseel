"""
api.py — FastAPI streaming backend for NIA
==========================================
Run with:  uvicorn api:app --reload --port 8000

The frontend connects to:
  POST /chat          → streaming SSE response
  GET  /health        → {"status": "ok"}
"""

import asyncio
import json
import warnings
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from src.graph import app as nia_graph

app = FastAPI(title="NIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


async def run_graph_and_stream(question: str) -> AsyncGenerator[str, None]:
    """
    Runs the LangGraph pipeline in a thread pool and streams:
      1. status events  → {"type": "status", "text": "..."}
      2. token events   → {"type": "token",  "text": "..."}
      3. done event     → {"type": "done"}
    """

    loop = asyncio.get_event_loop()

    # ── Step 1: yield a status while we figure out routing ───────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking…'})}\n\n"
    await asyncio.sleep(0)   # flush

    # Run the full graph in a thread (it's synchronous)
    initial_state = {
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
    }

    # We stream the graph using LangGraph's stream interface
    # Each yielded chunk is a node-output dict
    def _stream_graph():
        events = []
        for chunk in nia_graph.stream(initial_state, stream_mode="updates"):
            events.append(chunk)
        return events

    # We need to detect retrieval to show the right loading state.
    # We'll do a two-phase approach:
    #   Phase 1: run decide_retrieval synchronously (fast)
    #   Phase 2: stream the rest

    from src.nodes import decide_retrieval as _decide
    from src.state import MyState

    # Quick routing check
    decision_result = await loop.run_in_executor(None, lambda: _decide(initial_state))
    needs_retrieval = decision_result.get("need_retrieval", False)

    if needs_retrieval:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searching NADRA knowledge base…'})}\n\n"
        await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'status', 'text': 'Retrieving and grading documents…'})}\n\n"
        await asyncio.sleep(0)
    else:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Generating answer…'})}\n\n"
        await asyncio.sleep(0)

    # Run the full graph
    result = await loop.run_in_executor(None, lambda: nia_graph.invoke(initial_state))
    answer = result.get("answer", "No answer found.")

    # Stream the answer token by token (word-level for natural feel)
    words = answer.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        await asyncio.sleep(0.02)   # ~50 words/sec — feels natural

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_graph_and_stream(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}