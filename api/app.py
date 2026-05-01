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

# Per-language voice IDs from Uplift AI Orator docs (https://docs.upliftai.org/orator_voices)
# Falls back to UPLIFTAI_VOICE_ID (Urdu) for any unmapped language (e.g. English)
LANGUAGE_VOICE_MAP = {
    "urdu":    UPLIFTAI_VOICE_ID,       # v_8eelc901 — Info/Education Urdu
    "sindhi":  "v_sd6mn4p2",            # Male Calm Sindhi
    "balochi": "v_bl0ab8c4",            # Best Balochi male TTS
}


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Ensure conversation tables exist on startup."""
    try:
        await asyncio.to_thread(init_conversation_tables)
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


def _decide_retrieval_only(question: str) -> bool:
    """
    Run ONLY the decide_retrieval node synchronously — no full graph invocation.
    Used by the voice pipeline to route before deciding whether to translate,
    saving a GPT-4o-mini API call + ~500ms on every non-NADRA query.
    Falls back to True (retrieval) on any error — safer than skipping retrieval.
    """
    from src.routing import decision_llm
    from src.prompts import decide_retrieval_prompt
    try:
        decision = decision_llm.invoke(
            decide_retrieval_prompt.format_messages(question=question)
        )
        return decision.need_retrieval
    except Exception as e:
        logger.warning(f"[_decide_retrieval_only] Failed: {e} — defaulting to True")
        return True


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
    # Send conversation_id immediately so client stores it for subsequent turns.
    yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking…'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await asyncio.to_thread(
            _run_graph, question, "text", language, history
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
    await asyncio.to_thread(
        save_turn, conversation_id, question, answer, is_first
    )

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    language = request.language or "english"

    # Resolve or create backend conversation; fetch its stored history.
    if request.conversation_id:
        conv_id = request.conversation_id
        history = await asyncio.to_thread(get_conversation_history, conv_id)
    else:
        conv_id = await asyncio.to_thread(create_conversation, "text")
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
    yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"

    # ── Step 1: STT ───────────────────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Listening...'})}\n\n"
    await asyncio.sleep(0)

    try:
        suffix = "." + filename.split(".")[-1] if "." in filename else ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Whisper language hints — only pass codes the whisper-1 API actually accepts.
        # The open-source Whisper tokenizer defines "sd" for Sindhi, but the hosted
        # whisper-1 API endpoint rejects it with an unsupported language error.
        # Official well-performing languages per OpenAI docs include Urdu but not Sindhi.
        #
        #   "ur" → Urdu:    officially supported, correct hint
        #   None → Sindhi:  "sd" is rejected by the API — auto-detect performs better
        #                   than forcing "ur" (wrong phonology) or "sd" (rejected)
        #   None → Balochi: no Whisper model exists at all — auto-detect is the only option
        #   "en" → English: standard
        STT_LANG_MAP = {
            "urdu":    "ur",
            "sindhi":  None,   # "sd" rejected by whisper-1 API; auto-detect beats "ur"
            "balochi": None,   # no Balochi model; auto-detect beats a wrong hint
            "english": "en",
        }
        stt_lang = STT_LANG_MAP.get(language, "ur")

        with open(tmp_path, "rb") as audio_file:
            transcription = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                **({"language": stt_lang} if stt_lang is not None else {}),
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

    # ── Step 2: Route first — translate only if retrieval is needed ──────────
    # Calling decide_retrieval on the raw spoken text avoids paying for a
    # GPT-4o-mini translation on every greeting or general question ("hello",
    # "shukriya", etc.). Translation is expensive and adds ~500ms — skip it
    # whenever the graph would go straight to generate_direct anyway.
    yield f"data: {json.dumps({'type': 'status', 'text': 'Understanding your question...'})}\n\n"
    await asyncio.sleep(0)

    needs_retrieval = await asyncio.to_thread(_decide_retrieval_only, spoken_question)

    if needs_retrieval and _is_arabic_script(spoken_question):
        # Only translate when we are about to hit the vector store
        english_question = await translate_to_english(
            spoken_question, source_language=language.capitalize()
        )
    else:
        english_question = spoken_question
        if not needs_retrieval:
            logger.info("[TRANSLATE] Skipped — direct generation path, no retrieval needed")
        else:
            logger.info("[TRANSLATE] Skipped — no Arabic script detected")

    # ── Step 3: LangGraph pipeline ────────────────────────────────────────────
    yield f"data: {json.dumps({'type': 'status', 'text': 'Thinking...'})}\n\n"
    await asyncio.sleep(0)

    try:
        result = await asyncio.to_thread(
            _run_graph, english_question, "voice", language, history
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
        await asyncio.to_thread(
            save_turn, conversation_id, spoken_question, answer, is_first
        )

    try:
        if language == "english":
            # ── OpenAI TTS for English ────────────────────────────────────────
            tts_resp = await openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=tts_text,
                response_format="mp3",
            )
            audio_bytes_out = tts_resp.content
        else:
            # ── Uplift AI Orator for Urdu / Sindhi / Balochi ──────────────────
            async with httpx.AsyncClient(timeout=60) as client:
                tts_response = await client.post(
                    UPLIFTAI_TTS_URL,
                    headers={
                        "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "voiceId": LANGUAGE_VOICE_MAP.get(language, UPLIFTAI_VOICE_ID),
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
    audio_bytes = await file.read()
    logger.info(
        f"[Voice] file={file.filename} size={len(audio_bytes)} bytes "
        f"conv={conversation_id} lang={language}"
    )

    # Resolve or create backend conversation; fetch its stored history.
    if conversation_id:
        history = await asyncio.to_thread(get_conversation_history, conversation_id)
    else:
        conversation_id = await asyncio.to_thread(create_conversation, "voice")
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
    conversations = await asyncio.to_thread(list_conversations, limit)
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conv = await asyncio.to_thread(get_full_conversation, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/conversations/{conversation_id}")
async def remove_conversation(conversation_id: str):
    deleted = await asyncio.to_thread(delete_conversation, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "id": conversation_id}


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}