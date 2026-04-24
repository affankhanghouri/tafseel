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


# ── Translation helper ────────────────────────────────────────────────────────
async def translate_to_english(text: str) -> str:
    """
    Translate any Urdu text (Arabic script or Roman) into English.
    This ensures consistent input to the LangGraph pipeline regardless
    of how Whisper transcribed the audio.

    Returns the original text unchanged if translation fails.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translator. The user will give you a question in Urdu "
                        "(either Arabic script or Roman Urdu). Translate it into clear, "
                        "natural English. Keep all proper nouns exactly as-is: city names, "
                        "document names (CNIC, NICOP, B-Form, etc.), and place names. "
                        "Return ONLY the translated English question — no explanation, "
                        "no preamble, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            temperature=0,
        )
        translated = response.choices[0].message.content.strip()
        print(f"[TRANSLATE] '{text}' → '{translated}'")
        return translated
    except Exception as e:
        print(f"[TRANSLATE Error] {type(e).__name__}: {e} — using original text")
        return text


# ── Shared LangGraph runner ───────────────────────────────────────────────────
def _run_graph(question: str, mode: str = "text") -> dict:
    """Run LangGraph pipeline synchronously (called in thread pool)"""
    return nia_graph.invoke({
        "question":        question,
        "mode":            mode,
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

    result = await loop.run_in_executor(None, lambda: _run_graph(question, mode="text"))
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
      {"type": "transcript", "text": "<original transcribed question in Urdu>"}
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
        urdu_question = transcription.text.strip()

        if not urdu_question:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Could not understand audio. Please try again.'})}\n\n"
            return

        print(f"[STT] Transcribed: {urdu_question}")

    except Exception as e:
        print(f"[STT Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'text': f'Transcription failed: {str(e)}'})}\n\n"
        return

    # ── Step 2: Translate Urdu → English for reliable retrieval ──────────────
    # Whisper may output Arabic script Urdu which doesn't match the knowledge
    # base embeddings (ingested from English/Roman Urdu text files).
    # Translating to English first ensures consistent, high-quality retrieval.
    # The graph answer will still be generated in natural Urdu (mode="voice").
    yield f"data: {json.dumps({'type': 'status', 'text': 'Understanding your question...'})}\n\n"
    await asyncio.sleep(0)

    english_question = await translate_to_english(urdu_question)

    # ── Step 3: LangGraph pipeline (mode="voice" produces natural Urdu speech) ─
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking...'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await loop.run_in_executor(None, lambda: _run_graph(english_question, mode="voice"))
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

    # ── Step 4: TTS — UpliftAI converts Urdu answer to audio ─────────────────
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
            yield f"data: {json.dumps({'type': 'transcript', 'text': urdu_question})}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'text': f'TTS failed [{tts_response.status_code}]: {error_body}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        audio_bytes_out = tts_response.content
        print(f"[TTS] Success — audio size: {len(audio_bytes_out)} bytes")

        if len(audio_bytes_out) == 0:
            print("[TTS Error] Empty audio response")
            yield f"data: {json.dumps({'type': 'transcript', 'text': urdu_question})}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'text': 'TTS returned empty audio'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

    except httpx.TimeoutException:
        print("[TTS Error] Request timed out after 60s")
        yield f"data: {json.dumps({'type': 'transcript', 'text': urdu_question})}\n\n"
        yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'text': 'TTS failed: request timed out'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    except Exception as e:
        print(f"[TTS Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'transcript', 'text': urdu_question})}\n\n"
        yield f"data: {json.dumps({'type': 'answer', 'text': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'error', 'text': f'TTS failed: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # ── Step 5: Send transcript + answer + audio to Flutter ──────────────────
    # Note: transcript shows the original Urdu question (what the user said),
    # not the English translation — so the UI feels natural to the user.
    audio_b64 = base64.b64encode(audio_bytes_out).decode("utf-8")
    print(f"[TTS] Base64 length: {len(audio_b64)} chars — sending to Flutter")

    yield f"data: {json.dumps({'type': 'transcript', 'text': urdu_question})}\n\n"
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
            "Transfer-Encoding": "chunked",
        },
    )


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}