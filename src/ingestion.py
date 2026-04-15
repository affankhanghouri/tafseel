"""
ingestion.py — Parent-Child Chunking Pipeline for NIA
======================================================

Architecture
------------
  nadra_info.txt
       │
       ▼
  [ Parent Splitter ]  (1000 chars, 200 overlap)
       │
       ├──► SQLite  ← stores full parent text, keyed by UUID (parent_id)
       │
       ▼
  [ Child Splitter ]   (300 chars, 50 overlap)
       │
       ▼
  [ OpenAI Embeddings ]
       │
       ▼
  Qdrant Collection    ← stores child vectors + payload { parent_id, text, source }

Retrieval (ParentChildRetriever.invoke)
---------------------------------------
  query ──► embed ──► Qdrant top-k child hits
                           │
                           ▼
                    extract parent_id (deduplicate)
                           │
                           ▼
                    fetch parent text from SQLite
                           │
                           ▼
                    return List[Document]  (page_content = parent text)

The rest of the pipeline (nodes.py, graph.py) is unchanged — they
just call retriever.invoke(query) as before.
"""

import json
import logging
import os
import sqlite3
import uuid
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    RETRIEVER_K,
    SQLITE_PATH,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant client factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_qdrant_client() -> QdrantClient:
    """
    Returns a QdrantClient configured for:
      - Local file-based storage  (QDRANT_URL not set)
      - Self-hosted server        (QDRANT_URL=http://host:6333)
      - Qdrant Cloud              (QDRANT_URL=https://…qdrant.io + QDRANT_API_KEY)
    """
    if QDRANT_URL:
        logger.info("Connecting to Qdrant server at %s", QDRANT_URL)
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    logger.info("Using local Qdrant at path: %s", QDRANT_PATH)
    return QdrantClient(path=QDRANT_PATH)


def _ensure_collection(client: QdrantClient) -> None:
    """Creates the Qdrant collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", QDRANT_COLLECTION)


# ─────────────────────────────────────────────────────────────────────────────
# SQLite parent store
# ─────────────────────────────────────────────────────────────────────────────

def _make_sqlite_conn() -> sqlite3.Connection:
    """
    Opens (or creates) the SQLite database and ensures the parent_chunks
    table exists.  WAL journal mode is enabled for better write concurrency.
    """
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parent_chunks (
            id       TEXT PRIMARY KEY,
            text     TEXT NOT NULL,
            source   TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.commit()
    return conn


def _store_parent(
    conn: sqlite3.Connection,
    parent_id: str,
    text: str,
    source: str,
    metadata: dict,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO parent_chunks (id, text, source, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (parent_id, text, source, json.dumps(metadata)),
    )
    conn.commit()


def _fetch_parent(
    conn: sqlite3.Connection,
    parent_id: str,
) -> Optional[dict]:
    """Returns {text, source, metadata} or None if not found."""
    row = conn.execute(
        "SELECT text, source, metadata FROM parent_chunks WHERE id = ?",
        (parent_id,),
    ).fetchone()
    if row:
        return {
            "text": row[0],
            "source": row[1],
            "metadata": json.loads(row[2]),
        }
    return None


def _is_sqlite_empty(conn: sqlite3.Connection) -> bool:
    count = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0]
    return count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def ingest(source_path: str = "nadra_info.txt", batch_size: int = 100) -> None:
    """
    Full ingestion pipeline:
      1. Load raw text document.
      2. Split into parent chunks (large) → store in SQLite.
      3. Split each parent into child chunks (small) → embed → store in Qdrant.

    Args:
        source_path:  Path to the source .txt file.
        batch_size:   Number of Qdrant points to upsert per API call.
                      Larger batches are faster but use more memory.
    """
    logger.info("=== Starting ingestion from '%s' ===", source_path)

    # ── Load ──────────────────────────────────────────────────────────────────
    raw_docs = TextLoader(source_path, encoding="utf-8").load()
    logger.info("Loaded %d raw document(s)", len(raw_docs))

    # ── Splitters ─────────────────────────────────────────────────────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_docs = parent_splitter.split_documents(raw_docs)
    logger.info("Created %d parent chunks", len(parent_docs))

    # ── Stores ────────────────────────────────────────────────────────────────
    sqlite_conn   = _make_sqlite_conn()
    qdrant_client = _make_qdrant_client()
    _ensure_collection(qdrant_client)

    # ── Process each parent ───────────────────────────────────────────────────
    pending_points: List[PointStruct] = []
    total_children = 0

    for parent_doc in parent_docs:
        parent_id = str(uuid.uuid4())
        source    = parent_doc.metadata.get("source", source_path)

        # Store parent text in SQLite
        _store_parent(
            conn      = sqlite_conn,
            parent_id = parent_id,
            text      = parent_doc.page_content,
            source    = source,
            metadata  = parent_doc.metadata,
        )

        # Derive child chunks from this parent
        child_docs = child_splitter.split_documents([parent_doc])
        if not child_docs:
            continue

        # Embed all children in one batch call (minimises API round-trips)
        child_texts      = [c.page_content for c in child_docs]
        child_embeddings = embeddings.embed_documents(child_texts)

        for idx, (child_doc, child_vec) in enumerate(zip(child_docs, child_embeddings)):
            pending_points.append(
                PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = child_vec,
                    payload = {
                        "parent_id":   parent_id,
                        "text":        child_doc.page_content,
                        "source":      source,
                        "child_index": idx,
                    },
                )
            )
            total_children += 1

            # Flush to Qdrant when batch is full
            if len(pending_points) >= batch_size:
                qdrant_client.upsert(
                    collection_name = QDRANT_COLLECTION,
                    points          = pending_points,
                )
                logger.info(
                    "  Upserted batch of %d child points (running total: %d)",
                    len(pending_points),
                    total_children,
                )
                pending_points.clear()

    # Flush any remaining points
    if pending_points:
        qdrant_client.upsert(
            collection_name = QDRANT_COLLECTION,
            points          = pending_points,
        )

    sqlite_conn.close()
    qdrant_client.close()

    logger.info(
        "=== Ingestion complete: %d parents | %d children ===",
        len(parent_docs),
        total_children,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ParentChildRetriever
# ─────────────────────────────────────────────────────────────────────────────

class ParentChildRetriever:
    """
    Drop-in replacement for the old Chroma retriever.
    Exposes the same `.invoke(query) -> List[Document]` interface
    so nodes.py requires zero changes.

    Retrieval flow
    ──────────────
    1. Embed the query with OpenAI.
    2. Search Qdrant for the top-k nearest child chunks.
    3. Extract unique parent_ids from the hits.
    4. Fetch full parent texts from SQLite.
    5. Return List[Document] where page_content = parent text.
       (The LLM therefore always gets the broad, context-rich block,
        not just the narrow fragment that matched the query.)

    Metadata carried through
    ────────────────────────
    Each returned Document's metadata contains:
      - source      : originating file path
      - parent_id   : UUID linking back to SQLite row
      - child_text  : the exact child snippet that triggered the hit
      - score       : Qdrant cosine similarity score (0–1)
    """

    def __init__(self, k: int = RETRIEVER_K) -> None:
        self.k            = k
        self._qdrant      = _make_qdrant_client()
        self._sqlite      = _make_sqlite_conn()

    def invoke(self, query: str) -> List[Document]:
        if not query or not query.strip():
            return []

        # ── Embed query ───────────────────────────────────────────────────────
        query_vector = embeddings.embed_query(query)

        # ── Search Qdrant ─────────────────────────────────────────────────────
        hits = self._qdrant.query_points(
            collection_name = QDRANT_COLLECTION,
            query           = query_vector,
            limit           = self.k,
            with_payload    = True,
        ).points

        if not hits:
            logger.warning("Qdrant returned no results for query: %.80s", query)
            return []

        # ── Deduplicate by parent_id ──────────────────────────────────────────
        # Multiple child chunks from the same parent section may all rank in
        # the top-k.  We only need one copy of each parent.
        seen_parents: set = set()
        docs: List[Document] = []

        for hit in hits:
            parent_id  = hit.payload.get("parent_id", "")
            child_text = hit.payload.get("text", "")
            source     = hit.payload.get("source", "")
            score      = hit.score

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            # ── Fetch parent from SQLite ──────────────────────────────────────
            parent = _fetch_parent(self._sqlite, parent_id)

            if parent:
                docs.append(
                    Document(
                        page_content = parent["text"],
                        metadata     = {
                            "source":     source,
                            "parent_id":  parent_id,
                            "child_text": child_text,   # kept for debugging / logging
                            "score":      round(score, 4),
                        },
                    )
                )
            else:
                # Defensive fallback: parent missing from SQLite (should never
                # happen in a clean DB, but guards against partial ingestion).
                logger.warning("Parent '%s' not found in SQLite — returning child text", parent_id)
                docs.append(
                    Document(
                        page_content = child_text,
                        metadata     = {
                            "source":    source,
                            "parent_id": parent_id,
                            "score":     round(score, 4),
                        },
                    )
                )

        logger.debug(
            "Retrieved %d unique parent chunks for query: %.80s", len(docs), query
        )
        return docs

    def close(self) -> None:
        """Release DB connections.  Call this when shutting down the server."""
        self._qdrant.close()
        self._sqlite.close()

    # Allow use as a context manager
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap on import
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap(source_path: str = "nadra_info.txt") -> None:
    """
    Called once at module import time.
    Runs ingestion only if both stores are empty/missing.
    Safe to call repeatedly — it is idempotent.
    """
    sqlite_conn   = _make_sqlite_conn()
    qdrant_client = _make_qdrant_client()

    sqlite_empty  = _is_sqlite_empty(sqlite_conn)
    qdrant_empty  = QDRANT_COLLECTION not in {
        c.name for c in qdrant_client.get_collections().collections
    }

    sqlite_conn.close()
    qdrant_client.close()

    if sqlite_empty or qdrant_empty:
        logger.info("Stores are empty — running ingestion pipeline")
        ingest(source_path=source_path)
    else:
        logger.info("Existing stores detected — skipping ingestion (Qdrant + SQLite)")


_bootstrap()

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
# nodes.py imports this object and calls retriever.invoke(query).
# The interface is identical to the old Chroma retriever — no other file changes.

retriever = ParentChildRetriever(k=RETRIEVER_K)