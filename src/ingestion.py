"""
ingestion.py — Cloud Edition (Supabase + Qdrant Cloud)
======================================================
Supports multiple files in data/ folder.
Tracks ingested chunks by content hash — skips duplicates on re-run.
"""

import glob
import hashlib
import json
import logging
import os
import time
import uuid
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RETRIEVER_K,
    SUPABASE_DB_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Global connection pool for PostgreSQL (thread-safe)
_pg_pool = None


def _make_qdrant_client() -> QdrantClient:
    """Connects to Qdrant Cloud with increased timeout for uploads"""
    if QDRANT_URL:
        logger.info("Connecting to Qdrant Cloud...")
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
    logger.info("Using local Qdrant")
    return QdrantClient(path="qdrant_db")


def _ensure_collection(client: QdrantClient) -> None:
    """Creates collection if it doesn't exist"""
    try:
        client.get_collection(QDRANT_COLLECTION)
        logger.info(f"Collection '{QDRANT_COLLECTION}' exists")
    except Exception:
        logger.info(f"Creating collection '{QDRANT_COLLECTION}'...")
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Collection created successfully")
        except Exception as create_err:
            logger.error(f"Failed to create collection: {create_err}")
            raise


def _get_pg_pool():
    """Lazy initialization of PostgreSQL connection pool"""
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = SimpleConnectionPool(
            1, 5,
            SUPABASE_DB_URL,
            sslmode='require'
        )
    return _pg_pool


def _init_tables(conn) -> None:
    """Create all required tables if they don't exist"""
    with conn.cursor() as cur:
        # Parent chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                metadata JSONB NOT NULL DEFAULT '{}'
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent_chunks_id ON parent_chunks(id)"
        )
        # Ingestion tracking table — stores hash of each parent chunk
        # so we never re-ingest the same content twice
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingested_hashes (
                chunk_hash TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                ingested_at TIMESTAMP DEFAULT NOW()
            )
        """)
    conn.commit()


def _chunk_hash(text: str) -> str:
    """MD5 hash of chunk text — used to detect duplicates"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _is_already_ingested(conn, chunk_hash: str) -> bool:
    """Check if this chunk hash exists in tracking table"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM ingested_hashes WHERE chunk_hash = %s",
            (chunk_hash,)
        )
        return cur.fetchone() is not None


def _mark_as_ingested(conn, chunk_hash: str, source: str) -> None:
    """Record this hash as ingested"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingested_hashes (chunk_hash, source)
            VALUES (%s, %s)
            ON CONFLICT (chunk_hash) DO NOTHING
            """,
            (chunk_hash, source)
        )


def _store_parent(conn, parent_id: str, text: str, source: str, metadata: dict) -> None:
    """Store parent chunk in Supabase"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO parent_chunks (id, text, source, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                source = EXCLUDED.source,
                metadata = EXCLUDED.metadata
            """,
            (parent_id, text, source, json.dumps(metadata))
        )


def _fetch_parent(conn, parent_id: str) -> Optional[dict]:
    """Fetch parent from Supabase"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT text, source, metadata FROM parent_chunks WHERE id = %s",
            (parent_id,)
        )
        row = cur.fetchone()
        if row:
            return {
                "text": row["text"],
                "source": row["source"],
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
            }
        return None


def ingest_file(source_path: str, pg_conn, qdrant_client: QdrantClient, batch_size: int = 20) -> dict:
    """
    Ingest a single file. Skips chunks already ingested (by hash).
    Returns stats: {total, skipped, new}
    """
    logger.info(f"--- Ingesting: {source_path} ---")

    raw_docs = TextLoader(source_path, encoding="utf-8").load()

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
    logger.info(f"  {len(parent_docs)} parent chunks found")

    pending_points: List[PointStruct] = []
    total_children = 0
    skipped = 0
    new_parents = 0

    for i, parent_doc in enumerate(parent_docs):
        chunk_hash = _chunk_hash(parent_doc.page_content)

        # Skip if already ingested
        if _is_already_ingested(pg_conn, chunk_hash):
            skipped += 1
            continue

        parent_id = str(uuid.uuid4())
        source = parent_doc.metadata.get("source", source_path)

        # Store parent in Supabase
        _store_parent(pg_conn, parent_id, parent_doc.page_content, source, parent_doc.metadata)
        _mark_as_ingested(pg_conn, chunk_hash, source)
        new_parents += 1

        if i % 10 == 0:
            pg_conn.commit()
            logger.info(f"  Processed {i}/{len(parent_docs)} parents...")

        # Create and embed children
        child_docs = child_splitter.split_documents([parent_doc])
        if not child_docs:
            continue

        child_texts = [c.page_content for c in child_docs]
        child_embeddings = embeddings.embed_documents(child_texts)

        for idx, (child_doc, child_vec) in enumerate(zip(child_docs, child_embeddings)):
            pending_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=child_vec,
                    payload={
                        "parent_id": parent_id,
                        "text": child_doc.page_content,
                        "source": source,
                        "child_index": idx,
                    },
                )
            )
            total_children += 1

            if len(pending_points) >= batch_size:
                _upsert_with_retry(qdrant_client, pending_points)
                logger.info(f"  Upserted {total_children} children total...")
                pending_points.clear()

    # Flush remaining
    if pending_points:
        _upsert_with_retry(qdrant_client, pending_points)

    pg_conn.commit()

    stats = {
        "total": len(parent_docs),
        "skipped": skipped,
        "new": new_parents,
        "children": total_children,
    }
    logger.info(f"  Done: {new_parents} new | {skipped} skipped | {total_children} children embedded")
    return stats


def ingest_all(data_dir: str = "data", batch_size: int = 20) -> None:
    """
    Ingest all .txt files in data_dir.
    Skips chunks already ingested — safe to re-run at any time.
    """
    logger.info("=== Starting Multi-File Ingestion ===")

    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not txt_files:
        logger.warning(f"No .txt files found in '{data_dir}/'")
        return

    logger.info(f"Found {len(txt_files)} file(s): {[os.path.basename(f) for f in txt_files]}")

    pg_pool = _get_pg_pool()
    pg_conn = pg_pool.getconn()
    _init_tables(pg_conn)

    qdrant_client = _make_qdrant_client()
    _ensure_collection(qdrant_client)

    total_stats = {"total": 0, "skipped": 0, "new": 0, "children": 0}

    try:
        for file_path in txt_files:
            stats = ingest_file(file_path, pg_conn, qdrant_client, batch_size)
            for k in total_stats:
                total_stats[k] += stats[k]

        logger.info("=== Ingestion Complete ===")
        logger.info(f"  Total parents : {total_stats['total']}")
        logger.info(f"  New (ingested): {total_stats['new']}")
        logger.info(f"  Skipped (dupe): {total_stats['skipped']}")
        logger.info(f"  Children embedded: {total_stats['children']}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        pg_conn.rollback()
        raise
    finally:
        pg_pool.putconn(pg_conn)
        qdrant_client.close()


# ── Legacy single-file entry point (keeps backward compat) ──────────────────
def ingest(source_path: str = "data/nadra.txt", batch_size: int = 20) -> None:
    """Single-file ingest — wraps ingest_all for backward compatibility"""
    pg_pool = _get_pg_pool()
    pg_conn = pg_pool.getconn()
    _init_tables(pg_conn)
    qdrant_client = _make_qdrant_client()
    _ensure_collection(qdrant_client)
    try:
        ingest_file(source_path, pg_conn, qdrant_client, batch_size)
    except Exception as e:
        pg_conn.rollback()
        raise
    finally:
        pg_pool.putconn(pg_conn)
        qdrant_client.close()


def _upsert_with_retry(client: QdrantClient, points: List[PointStruct], max_retries=3):
    """Retry upsert with exponential backoff for cloud timeouts"""
    for attempt in range(max_retries):
        try:
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points,
                wait=True
            )
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (attempt + 1) * 2
            logger.warning(f"Upsert failed, retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)


class ParentChildRetriever:
    """
    Retrieves from Qdrant Cloud + Supabase PostgreSQL.
    """

    def __init__(self, k: int = RETRIEVER_K) -> None:
        self.k = k
        self._qdrant = None
        self._pg_pool = None
        self._init_connections()

    def _init_connections(self):
        try:
            self._qdrant = _make_qdrant_client()
            self._pg_pool = _get_pg_pool()
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise

    def invoke(self, query: str) -> List[Document]:
        if not query or not query.strip():
            return []

        if not self._qdrant:
            self._init_connections()

        query_vector = embeddings.embed_query(query)

        hits = self._qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=self.k,
            with_payload=True,
        ).points

        if not hits:
            logger.warning(f"No Qdrant results for: {query[:80]}...")
            return []

        seen_parents = set()
        docs = []
        pg_conn = self._pg_pool.getconn()

        try:
            for hit in hits:
                parent_id = hit.payload.get("parent_id", "")
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent = _fetch_parent(pg_conn, parent_id)
                if parent:
                    docs.append(
                        Document(
                            page_content=parent["text"],
                            metadata={
                                "source": parent["source"],
                                "parent_id": parent_id,
                                "child_text": hit.payload.get("text", ""),
                                "score": round(hit.score, 4),
                            },
                        )
                    )
                else:
                    docs.append(
                        Document(
                            page_content=hit.payload.get("text", ""),
                            metadata={
                                "source": hit.payload.get("source", ""),
                                "parent_id": parent_id,
                                "score": round(hit.score, 4),
                            },
                        )
                    )
        finally:
            self._pg_pool.putconn(pg_conn)

        return docs

    def close(self):
        if self._qdrant:
            self._qdrant.close()


# LAZY INITIALIZATION
retriever = None


def get_retriever():
    """Get or create retriever singleton"""
    global retriever
    if retriever is None:
        retriever = ParentChildRetriever(k=RETRIEVER_K)
    return retriever