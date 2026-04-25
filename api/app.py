"""
app.py — FastAPI streaming backend for NIA
==========================================
Run with:  uvicorn app:app --reload --port 8000

Endpoints:
  POST /chat                        → streaming SSE (text)
  POST /voice                       → streaming SSE (status updates + base64 audio)
  GET  /conversations               → list all past conversations
  GET  /conversations/{id}          → full conversation with all messages
  DELETE /conversations/{id}        → delete a conversation
  GET  /health                      → {"status": "ok"}
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import warnings
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.graph import app as nia_graph
from src.conversation_store import (
    init_conversation_tables,
    create_conversation,
    save_turn,
    get_conversation_history,
    list_conversations,
    get_full_conversation,
    delete_conversation,
)

app = FastAPI(title="NIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ───────────────────────────────────────────────────────────────────
openai_client   = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
UPLIFTAI_API_KEY  = os.getenv("UPLIFTAI_API_KEY")
UPLIFTAI_TTS_URL  = "https://api.upliftai.org/v1/synthesis/text-to-speech"
UPLIFTAI_VOICE_ID = os.getenv("UPLIFTAI_VOICE_ID", "v_8eelc901")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Ensure conversation tables exist on startup."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_conversation_tables)
        logger.info("✅ Conversation tables initialised")
    except Exception as e:
        logger.error(f"⚠️  Could not init conversation tables: {e}")

    if not UPLIFTAI_API_KEY:
        logger.warning("⚠️  UPLIFTAI_API_KEY not set — /voice TTS will fail")
    else:
        logger.info(f"✅ UPLIFTAI_API_KEY loaded: {UPLIFTAI_API_KEY[:8]}...")


# ── Shared helpers ────────────────────────────────────────────────────────────
def _is_arabic_script(text: str) -> bool:
    return any('\u0600' <= c <= '\u06FF' for c in text)


async def translate_to_english(text: str, source_language: str = "Urdu") -> str:
    """Translate user question → English for reliable retrieval."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a translator. The user will give you a question in {source_language} "
                        f"(either Arabic script or Roman {source_language}). Translate it into clear, "
                        f"natural English. Keep all proper nouns exactly as-is: city names, "
                        f"document names (CNIC, NICOP, B-Form, etc.), and place names. "
                        f"Return ONLY the translated English question — no explanation, "
                        f"no preamble, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            temperature=0,
        )
        translated = response.choices[0].message.content.strip()
        logger.info(f"[TRANSLATE] '{text[:60]}' → '{translated[:60]}'")
        return translated
    except Exception as e:
        logger.error(f"[TRANSLATE Error] {type(e).__name__}: {e} — using original text")
        return text


def _run_graph(question: str, mode: str, language: str, history: list) -> dict:
    """Run LangGraph pipeline synchronously (called in thread pool)."""
    return nia_graph.invoke({
        "question":         question,
        "mode":             mode,
        # ── FIX: pass language so nodes generate in the correct language ───────
        "language":         language,
        "conversation_id":  None,
        # ── FIX: pass history so the LLM has conversational context ───────────
        "history":          history,
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


# ── /chat endpoint ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    language: Optional[str] = "english"   # chat is always English for now


async def run_graph_and_stream(
    question: str,
    conversation_id: str,
    language: str,
    history: list,
) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()

    # Send conversation_id immediately so client stores it for subsequent turns.
    yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking…'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: _run_graph(question, "text", language, history),
        )
    except Exception as e:
        logger.error(f"[/chat graph error] {e}")
        yield f"data: {json.dumps({'type': 'error', 'text': 'Something went wrong. Please try again.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    needs_retrieval = result.get("need_retrieval", False)
    if needs_retrieval:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searched NADRA knowledge base'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Generated answer'})}\n\n"
    await asyncio.sleep(0)

    answer = result.get("answer", "No answer found.")

    # Stream tokens word by word.
    words = answer.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else " " + word
        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        await asyncio.sleep(0.02)

    # Save BEFORE yielding done so it always executes even if client disconnects.
    is_first = len(history) == 0
    await loop.run_in_executor(
        None,
        lambda: save_turn(conversation_id, question, answer, is_first_turn=is_first),
    )

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    loop = asyncio.get_event_loop()
    language = request.language or "english"

    # Resolve or create backend conversation; fetch its stored history.
    if request.conversation_id:
        conv_id = request.conversation_id
        history = await loop.run_in_executor(
            None, lambda: get_conversation_history(conv_id)
        )
    else:
        conv_id = await loop.run_in_executor(
            None, lambda: create_conversation(mode="text")
        )
        history = []

    return StreamingResponse(
        run_graph_and_stream(request.question, conv_id, language, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /voice endpoint ───────────────────────────────────────────────────────────
async def run_voice_pipeline(
    audio_bytes: bytes,
    filename: str,
    conversation_id: str,
    language: str,
    history: list,
) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()

    yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"

    # ── Step 1: STT ───────────────────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Listening...'})}\n\n"
    await asyncio.sleep(0)

    try:
        suffix = "." + filename.split(".")[-1] if "." in filename else ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # FIX: Whisper handles Arabic-script audio best when hinted with "ur".
        # Sindhi and Balochi are also Arabic-script, so we use "ur" for all three.
        stt_lang = "ur" if language in ("urdu", "sindhi", "balochi") else "en"

        with open(tmp_path, "rb") as audio_file:
            transcription = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=stt_lang,
            )
        os.unlink(tmp_path)
        spoken_question = transcription.text.strip()

        if not spoken_question:
            yield f"data: {json.dumps({'type': 'error', 'text': 'Could not understand audio. Please try again.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        logger.info(f"[STT] Transcribed: {spoken_question}")

    except Exception as e:
        logger.error(f"[STT Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'text': f'Transcription failed: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # ── Step 2: Translate to English only if needed for retrieval ─────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Understanding your question...'})}\n\n"
    await asyncio.sleep(0)

    if _is_arabic_script(spoken_question):
        # FIX: pass the real source language so the translator doesn't assume Urdu
        english_question = await translate_to_english(
            spoken_question, source_language=language.capitalize()
        )
    else:
        english_question = spoken_question
        logger.info("[TRANSLATE] Skipped — no Arabic script detected")

    # ── Step 3: LangGraph pipeline ────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking...'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: _run_graph(english_question, "voice", language, history),
        )
    except Exception as e:
        logger.error(f"[Graph Error] {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'text': f'Pipeline failed: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    needs_retrieval = result.get("need_retrieval", False)
    if needs_retrieval:
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searching NADRA knowledge base...'})}\n\n"
        await asyncio.sleep(0.1)

    answer = result.get("answer", "No answer found.")
    logger.info(f"[Graph] Answer ({len(answer)} chars): {answer[:120]}...")

    # ── Step 4: TTS ───────────────────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Preparing answer...'})}\n\n"
    await asyncio.sleep(0)

    MAX_TTS_CHARS = 3000
    tts_text = answer[:MAX_TTS_CHARS]

    async def _save_turn():
        is_first = len(history) == 0
        await loop.run_in_executor(
            None,
            lambda: save_turn(conversation_id, spoken_question, answer, is_first_turn=is_first),
        )

    try:
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

        if tts_response.status_code != 200:
            error_body = tts_response.text
            logger.error(f"[TTS Error] Status {tts_response.status_code} | {error_body}")
            yield f"data: {json.dumps({'type': 'transcript', 'text': spoken_question})}\n\n"
            yield f"data: {json.dumps({'type': 'answer',     'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'error',      'text': f'TTS failed [{tts_response.status_code}]'})}\n\n"
            await _save_turn()
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        audio_bytes_out = tts_response.content
        if len(audio_bytes_out) == 0:
            yield f"data: {json.dumps({'type': 'transcript', 'text': spoken_question})}\n\n"
            yield f"data: {json.dumps({'type': 'answer',     'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'error',      'text': 'TTS returned empty audio'})}\n\n"
            await _save_turn()
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

    except httpx.TimeoutException:
        yield f"data: {json.dumps({'type': 'transcript', 'text': spoken_question})}\n\n"
        yield f"data: {json.dumps({'type': 'answer',     'text': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'error',      'text': 'TTS failed: request timed out'})}\n\n"
        await _save_turn()
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    except Exception as e:
        yield f"data: {json.dumps({'type': 'transcript', 'text': spoken_question})}\n\n"
        yield f"data: {json.dumps({'type': 'answer',     'text': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'error',      'text': f'TTS failed: {str(e)}'})}\n\n"
        await _save_turn()
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # ── Step 5: Send everything to client ─────────────────────────────────────
    audio_b64 = base64.b64encode(audio_bytes_out).decode("utf-8")
    yield f"data: {json.dumps({'type': 'transcript', 'text': spoken_question})}\n\n"
    yield f"data: {json.dumps({'type': 'answer',     'text': answer})}\n\n"
    yield f"data: {json.dumps({'type': 'audio',      'data': audio_b64})}\n\n"

    await _save_turn()

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/voice")
async def voice(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(default=None),
    language: str = Form(default="urdu"),
):
    loop = asyncio.get_event_loop()
    audio_bytes = await file.read()
    logger.info(
        f"[Voice] file={file.filename} size={len(audio_bytes)} bytes "
        f"conv={conversation_id} lang={language}"
    )

    # Resolve or create backend conversation; fetch its stored history.
    if conversation_id:
        history = await loop.run_in_executor(
            None, lambda: get_conversation_history(conversation_id)
        )
    else:
        conversation_id = await loop.run_in_executor(
            None, lambda: create_conversation(mode="voice")
        )
        history = []

    return StreamingResponse(
        run_voice_pipeline(
            audio_bytes,
            file.filename or "audio.m4a",
            conversation_id,
            language,
            history,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


# ── Conversation history endpoints ────────────────────────────────────────────
@app.get("/conversations")
async def get_conversations(limit: int = 50):
    loop = asyncio.get_event_loop()
    conversations = await loop.run_in_executor(None, lambda: list_conversations(limit))
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    loop = asyncio.get_event_loop()
    conv = await loop.run_in_executor(None, lambda: get_full_conversation(conversation_id))
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/conversations/{conversation_id}")
async def remove_conversation(conversation_id: str):
    loop = asyncio.get_event_loop()
    deleted = await loop.run_in_executor(None, lambda: delete_conversation(conversation_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "id": conversation_id}


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}