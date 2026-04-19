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
UPLIFTAI_VOICE_ID = os.getenv("UPLIFTAI_VOICE_ID", "v_8eelc901")  # Urdu Info/Education

# ── Startup key check ─────────────────────────────────────────────────────────
if not UPLIFTAI_API_KEY:
    print("⚠️  WARNING: UPLIFTAI_API_KEY is not set in .env — /voice endpoint will fail")
else:
    print(f"✅ UPLIFTAI_API_KEY loaded: {UPLIFTAI_API_KEY[:8]}...")


# ── Shared LangGraph runner ───────────────────────────────────────────────────
def _run_graph(question: str, urdu_mode: bool = False) -> dict:
    """Run LangGraph pipeline synchronously (called in thread pool)"""
    # For voice: inject Urdu instruction into question so LLM answers in Urdu
    if urdu_mode:
        augmented_question = (
            f"{question}\n\n"
            "[IMPORTANT: Answer MUST be in Urdu script (اردو). "
            "Do NOT use English. Write naturally in Urdu as if speaking to a Pakistani citizen.]"
        )
    else:
        augmented_question = question

    return nia_graph.invoke({
        "question":        augmented_question,
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


# ── /chat endpoint ────────────────────────────────────────────────────────────
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


# ── /voice endpoint ───────────────────────────────────────────────────────────
async def run_voice_pipeline(audio_bytes: bytes, filename: str) -> AsyncGenerator[str, None]:
    """
    Streams status events then final audio:
      {"type": "status",     "text": "Listening..."}
      {"type": "status",     "text": "Thinking..."}
      {"type": "status",     "text": "Searching NADRA knowledge base..."}  ← only if retrieval
      {"type": "status",     "text": "Preparing answer..."}
      {"type": "transcript", "text": "<transcribed question>"}
      {"type": "answer",     "text": "<answer text in Urdu>"}
      {"type": "audio",      "data": "<base64 encoded mp3>"}
      {"type": "done"}
    """
    loop = asyncio.get_event_loop()

    # ── Step 1: STT — Whisper transcribes audio ───────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Listening...'})}\n\n"
    await asyncio.sleep(0)

    try:
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

        print(f"[STT] Transcribed: {question}")

    except Exception as e:
        print(f"[STT Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'text': f'Transcription failed: {str(e)}'})}\n\n"
        return

    # ── Step 2: LangGraph pipeline (urdu_mode=True forces Urdu answers) ───────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking...'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await loop.run_in_executor(None, lambda: _run_graph(question, urdu_mode=True))
    except Exception as e:
        print(f"[Graph Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'text': f'Pipeline failed: {str(e)}'})}\n\n"
        return

    # ── Tell Flutter if retrieval happened so it can show a searching state ───
    needs_retrieval = result.get("need_retrieval", False)
    if needs_retrieval:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searching NADRA knowledge base...'})}\n\n"
        await asyncio.sleep(0.1)

    answer = result.get("answer", "No answer found.")
    print(f"[Graph] Answer ({len(answer)} chars): {answer[:120]}...")

    # ── Sanitise answer: strip the injected instruction if LLM echoed it ─────
    # (some models repeat the system instruction in their output)
    if "[IMPORTANT:" in answer:
        answer = answer.split("[IMPORTANT:")[0].strip()

    # ── Step 3: TTS — UpliftAI converts Urdu answer to audio ─────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Preparing answer...'})}\n\n"
    await asyncio.sleep(0)

    # Truncate very long answers to stay within UpliftAI free-tier limits
    # (~10 min of audio; ~3000 Urdu chars is roughly 5 min)
    MAX_TTS_CHARS = 3000
    tts_text = answer[:MAX_TTS_CHARS]
    if len(answer) > MAX_TTS_CHARS:
        print(f"[TTS] Answer truncated from {len(answer)} to {MAX_TTS_CHARS} chars for TTS")

    try:
        print(f"[TTS] Calling UpliftAI | voice={UPLIFTAI_VOICE_ID} | text_len={len(tts_text)}")

        async with httpx.AsyncClient(timeout=60) as client:
            tts_response = await client.post(
                UPLIFTAI_TTS_URL,
                headers={
                    "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "voiceId": UPLIFTAI_VOICE_ID,
                    "text": tts_text,
                    "outputFormat": "MP3_22050_32",
                },
            )

        print(f"[TTS] Response status: {tts_response.status_code}")
        print(f"[TTS] Content-Type: {tts_response.headers.get('content-type', 'unknown')}")

        if tts_response.status_code != 200:
            error_body = tts_response.text
            print(f"[TTS Error] Status {tts_response.status_code} | Body: {error_body}")
            # Still send the answer text even if TTS fails
            yield f"data: {json.dumps({'type': 'transcript', 'text': question})}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'text': f'TTS failed [{tts_response.status_code}]: {error_body}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        audio_bytes_out = tts_response.content
        print(f"[TTS] Success — audio size: {len(audio_bytes_out)} bytes")

        if len(audio_bytes_out) == 0:
            print("[TTS Error] Empty audio response")
            yield f"data: {json.dumps({'type': 'transcript', 'text': question})}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'text': 'TTS returned empty audio'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

    except httpx.TimeoutException:
        print("[TTS Error] Request timed out after 60s")
        yield f"data: {json.dumps({'type': 'transcript', 'text': question})}\n\n"
        yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'text': 'TTS failed: request timed out'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    except Exception as e:
        print(f"[TTS Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'transcript', 'text': question})}\n\n"
        yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'text': f'TTS failed: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # ── Step 4: Send transcript + answer + audio to Flutter ──────────────────
    audio_b64 = base64.b64encode(audio_bytes_out).decode("utf-8")
    print(f"[TTS] Base64 length: {len(audio_b64)} chars — sending to Flutter")

    yield f"data: {json.dumps({'type': 'transcript', 'text': question})}\n\n"
    yield f"data: {json.dumps({'type': 'answer',     'text': answer})}\n\n"
    yield f"data: {json.dumps({'type': 'audio',      'data': audio_b64})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    """
    Accepts an audio file from Flutter.
    Returns SSE stream with status updates and final base64 audio.
    """
    audio_bytes = await file.read()
    print(f"[Voice] Received file: {file.filename} | size: {len(audio_bytes)} bytes")
    return StreamingResponse(
        run_voice_pipeline(audio_bytes, file.filename or "audio.m4a"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            # Disable any proxy buffering — critical for large SSE events
            "Transfer-Encoding": "chunked",
        },
    )


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}