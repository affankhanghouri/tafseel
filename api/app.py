"""
app.py — FastAPI streaming backend for NIA
==========================================
Run with:  uvicorn app:app --reload --port 8000

Endpoints:
  POST /chat          → streaming SSE (text)
  POST /voice         → streaming SSE (status updates + base64 audio)
  GET  /health        → {"status": "ok"}
"""

import asyncio
import base64
import json
import os
import tempfile
import warnings
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
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

# ── Clients ──────────────────────────────────────────────────────────────────
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
UPLIFTAI_API_KEY = os.getenv("UPLIFTAI_API_KEY")
UPLIFTAI_TTS_URL = "https://api.upliftai.org/v1/synthesis/text-to-speech"
UPLIFTAI_VOICE_ID = "v_meklc281"  # Natural Pakistani Urdu voice


# ── Shared LangGraph runner ───────────────────────────────────────────────────
def _run_graph(question: str) -> dict:
    """Run LangGraph pipeline synchronously (called in thread pool)"""
    return nia_graph.invoke({
        "question":        question,
        "need_retrieval":  False,
        "docs":            [],
        "relevant_docs":   [],
        "context":         "",
        "answer":          "",
        "retrieval_query": "",
        "retries":         0,
        "rewrite_tries":   0,
        "issup":           "",
        "evidence":        [],
        "is_useful":       False,
    })


# ── /chat endpoint (existing - unchanged) ────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


async def run_graph_and_stream(question: str) -> AsyncGenerator[str, None]:
    """
    Streams:
      1. status events  → {"type": "status", "text": "..."}
      2. token events   → {"type": "token",  "text": "..."}
      3. done event     → {"type": "done"}
    """
    loop = asyncio.get_event_loop()

    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking…'})}\n\n"
    await asyncio.sleep(0)

    result = await loop.run_in_executor(None, lambda: _run_graph(question))
    needs_retrieval = result.get("need_retrieval", False)

    if needs_retrieval:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searched NADRA knowledge base'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Generated answer'})}\n\n"
    await asyncio.sleep(0)

    answer = result.get("answer", "No answer found.")

    words = answer.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        await asyncio.sleep(0.02)

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


# ── /voice endpoint (new) ─────────────────────────────────────────────────────
async def run_voice_pipeline(audio_bytes: bytes, filename: str) -> AsyncGenerator[str, None]:
    """
    Streams status events then final audio:
      {"type": "status", "text": "Listening..."}
      {"type": "status", "text": "Thinking..."}
      {"type": "status", "text": "Searching NADRA knowledge base..."}  ← only if retrieval
      {"type": "status", "text": "Preparing answer..."}
      {"type": "audio",  "data": "<base64 encoded mp3>"}
      {"type": "done"}
    """
    loop = asyncio.get_event_loop()

    # ── Step 1: STT — Whisper transcribes audio ───────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Listening...'})}\n\n"
    await asyncio.sleep(0)

    try:
        # Save audio bytes to a temp file for Whisper
        suffix = "." + filename.split(".")[-1] if "." in filename else ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            transcription = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ur",
            )
        os.unlink(tmp_path)
        question = transcription.text.strip()

        if not question:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Could not understand audio. Please try again.'})}\n\n"
            return

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'text': f'Transcription failed: {str(e)}'})}\n\n"
        return

    # ── Step 2: LangGraph pipeline ────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking...'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await loop.run_in_executor(None, lambda: _run_graph(question))
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'text': f'Pipeline failed: {str(e)}'})}\n\n"
        return

    needs_retrieval = result.get("need_retrieval", False)
    if needs_retrieval:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searching NADRA knowledge base...'})}\n\n"
        await asyncio.sleep(0)

    answer = result.get("answer", "No answer found.")

    # ── Step 3: TTS — UpliftAI converts answer to Urdu audio ─────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Preparing answer...'})}\n\n"
    await asyncio.sleep(0)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            tts_response = await client.post(
                UPLIFTAI_TTS_URL,
                headers={
                    "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "voiceId": UPLIFTAI_VOICE_ID,
                    "text": answer,
                    "outputFormat": "MP3_22050_32",
                },
            )
            tts_response.raise_for_status()
            audio_bytes_out = tts_response.content

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'text': f'TTS failed: {str(e)}'})}\n\n"
        return

    # ── Step 4: Send audio as base64 to Flutter ───────────────────────────────
    audio_b64 = base64.b64encode(audio_bytes_out).decode("utf-8")
    yield f"data: {json.dumps({'type': 'transcript', 'text': question})}\n\n"
    yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    """
    Accepts an audio file from Flutter.
    Returns SSE stream with status updates and final base64 audio.
    
    Flutter sends:
      MultipartRequest with field name 'file'
      Supported formats: m4a, mp3, wav, webm, ogg
    """
    audio_bytes = await file.read()
    return StreamingResponse(
        run_voice_pipeline(audio_bytes, file.filename or "audio.m4a"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}