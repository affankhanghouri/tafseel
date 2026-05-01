"""
conversation_store.py — Conversation persistence layer (Supabase)
=================================================================
Handles creating, reading, and updating conversations and messages.
Imported by app.py — completely separate from the RAG ingestion logic.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from src.config import SUPABASE_DB_URL

logger = logging.getLogger(__name__)

_pg_pool = None
MAX_TITLE_LENGTH = 60
HISTORY_WINDOW = 5    # max TURNS (user+assistant pairs) fetched for LLM context
# Each turn = 2 messages, so we fetch HISTORY_WINDOW * 2 from the DB.
# Turn-based counting ensures a single long message cannot displace recent turns.


def _get_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = SimpleConnectionPool(1, 5, SUPABASE_DB_URL, sslmode="require")
    return _pg_pool


def init_conversation_tables() -> None:
    """Create conversations + messages tables if they don't exist."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New Conversation',
                    mode        TEXT NOT NULL DEFAULT 'text',
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id               TEXT PRIMARY KEY,
                    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role             TEXT NOT NULL,   -- 'user' or 'assistant'
                    content          TEXT NOT NULL,
                    created_at       TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conv_id
                ON messages(conversation_id, created_at ASC)
            """)
        conn.commit()
        logger.info("Conversation tables ready")
    except Exception as e:
        conn.rollback()
        logger.error(f"[init_conversation_tables] {e}")
        raise
    finally:
        pool.putconn(conn)


def create_conversation(mode: str = "text", title: str = "New Conversation") -> str:
    """Create a new conversation row. Returns conversation_id."""
    pool = _get_pool()
    conn = pool.getconn()
    conv_id = str(uuid.uuid4())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, title, mode) VALUES (%s, %s, %s)",
                (conv_id, title[:MAX_TITLE_LENGTH], mode),
            )
        conn.commit()
        return conv_id
    except Exception as e:
        conn.rollback()
        logger.error(f"[create_conversation] {e}")
        raise
    finally:
        pool.putconn(conn)


def _auto_title(question: str) -> str:
    """Generate a conversation title from the first user question."""
    title = question.strip()
    if len(title) > MAX_TITLE_LENGTH:
        title = title[:MAX_TITLE_LENGTH - 1] + "…"
    return title or "New Conversation"


def save_turn(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    is_first_turn: bool = False,
) -> None:
    """
    Append one user + one assistant message to the conversation.
    If is_first_turn=True, also update the conversation title from the question.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            # Insert user message
            cur.execute(
                "INSERT INTO messages (id, conversation_id, role, content) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), conversation_id, "user", user_message),
            )
            # Insert assistant message
            cur.execute(
                "INSERT INTO messages (id, conversation_id, role, content) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), conversation_id, "assistant", assistant_message),
            )
            # Update conversation updated_at (and title on first turn)
            if is_first_turn:
                cur.execute(
                    "UPDATE conversations SET updated_at = NOW(), title = %s WHERE id = %s",
                    (_auto_title(user_message), conversation_id),
                )
            else:
                cur.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                    (conversation_id,),
                )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"[save_turn] {e}")
        # Don't raise — a save failure should not crash the response
    finally:
        pool.putconn(conn)


def get_conversation_history(conversation_id: str, max_turns: int = HISTORY_WINDOW) -> List[dict]:
    """
    Fetch the last `max_turns` complete turns (user+assistant pairs).
    Returns list of {role, content} dicts ordered oldest→newest.
    Turn-based limit prevents a single verbose message from crowding out recent turns.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub ORDER BY created_at ASC
                """,
                (conversation_id, max_turns * 2),   # 2 messages per turn
            )
            rows = cur.fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception as e:
        logger.error(f"[get_conversation_history] {e}")
        return []
    finally:
        pool.putconn(conn)


def list_conversations(limit: int = 50) -> List[dict]:
    """Return the most recent conversations (newest first)."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.id, c.title, c.mode, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "mode": row["mode"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "message_count": row["message_count"],
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"[list_conversations] {e}")
        return []
    finally:
        pool.putconn(conn)


def get_full_conversation(conversation_id: str) -> Optional[dict]:
    """Return conversation metadata + all messages."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, mode, created_at, updated_at FROM conversations WHERE id = %s",
                (conversation_id,),
            )
            conv = cur.fetchone()
            if not conv:
                return None

            cur.execute(
                "SELECT id, role, content, created_at FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
                (conversation_id,),
            )
            messages = cur.fetchall()

        return {
            "id": conv["id"],
            "title": conv["title"],
            "mode": conv["mode"],
            "created_at": conv["created_at"].isoformat(),
            "updated_at": conv["updated_at"].isoformat(),
            "messages": [
                {
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "created_at": m["created_at"].isoformat(),
                }
                for m in messages
            ],
        }
    except Exception as e:
        logger.error(f"[get_full_conversation] {e}")
        return None
    finally:
        pool.putconn(conn)


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages (CASCADE). Returns True if deleted."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        logger.error(f"[delete_conversation] {e}")
        return False
    finally:
        pool.putconn(conn)