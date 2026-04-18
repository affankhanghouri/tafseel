"""
ingestion.py — Cloud Edition (Supabase + Qdrant Cloud)
======================================================
Supports multiple files in data/ folder.
Tracks ingested files by whole-file hash.
If file unchanged → skip entirely.
If file changed → delete old vectors → re-ingest fresh.
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
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

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

_pg_pool = None


def _make_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        logger.info("Connecting to Qdrant Cloud...")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    logger.info("Using local Qdrant")
    return QdrantClient(path="qdrant_db")


def _ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(QDRANT_COLLECTION)
        logger.info(f"Collection '{QDRANT_COLLECTION}' exists")
    except Exception:
        logger.info(f"Creating collection '{QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
        )
        logger.info("Collection created successfully")


def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = SimpleConnectionPool(1, 5, SUPABASE_DB_URL, sslmode='require')
    return _pg_pool


def _init_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                metadata JSONB NOT NULL DEFAULT '{}'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parent_chunks_id ON parent_chunks(id)")

        # File-level hash tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingested_files (
                source TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                ingested_at TIMESTAMP DEFAULT NOW()
            )
        """)
    conn.commit()


def _file_hash(file_path: str) -> str:
    """MD5 hash of entire file content"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _get_stored_hash(conn, source: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT file_hash FROM ingested_files WHERE source = %s", (source,))
        row = cur.fetchone()
        return row[0] if row else None


def _set_stored_hash(conn, source: str, file_hash: str) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ingested_files (source, file_hash)
            VALUES (%s, %s)
            ON CONFLICT (source) DO UPDATE SET
                file_hash = EXCLUDED.file_hash,
                ingested_at = NOW()
        """, (source, file_hash))


def _delete_file_data(conn, qdrant_client: QdrantClient, source: str) -> None:
    """Delete all vectors and parent chunks for a given source file"""
    logger.info(f"  Deleting old data for: {source}")

    # Delete from Qdrant by source filter
    qdrant_client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )

    # Delete from Supabase
    with conn.cursor() as cur:
        cur.execute("DELETE FROM parent_chunks WHERE source = %s", (source,))

    conn.commit()
    logger.info(f"  Old data deleted for: {source}")


def _store_parent(conn, parent_id: str, text: str, source: str, metadata: dict) -> None:
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
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT text, source, metadata FROM parent_chunks WHERE id = %s", (parent_id,))
        row = cur.fetchone()
        if row:
            return {
                "text": row["text"],
                "source": row["source"],
                "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"])
            }
        return None


def ingest_file(file_path: str, pg_conn, qdrant_client: QdrantClient, batch_size: int = 20) -> str:
    """
    Ingest a single file using file-level hash tracking.
    Returns: 'skipped', 'new', or 'updated'
    """
    source = file_path
    current_hash = _file_hash(file_path)
    stored_hash = _get_stored_hash(pg_conn, source)

    if stored_hash == current_hash:
        logger.info(f"  Skipping (unchanged): {file_path}")
        return 'skipped'

    if stored_hash is not None:
        # File changed — delete old data first
        logger.info(f"  File changed: {file_path} — re-ingesting...")
        _delete_file_data(pg_conn, qdrant_client, source)
        status = 'updated'
    else:
        logger.info(f"  New file: {file_path} — ingesting...")
        status = 'new'

    raw_docs = TextLoader(file_path, encoding="utf-8").load()

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

    for i, parent_doc in enumerate(parent_docs):
        parent_id = str(uuid.uuid4())
        parent_doc.metadata["source"] = source
        _store_parent(pg_conn, parent_id, parent_doc.page_content, source, parent_doc.metadata)

        if i % 10 == 0:
            pg_conn.commit()
            logger.info(f"  Processed {i}/{len(parent_docs)} parents...")

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

    if pending_points:
        _upsert_with_retry(qdrant_client, pending_points)

    _set_stored_hash(pg_conn, source, current_hash)
    pg_conn.commit()

    logger.info(f"  Done: {len(parent_docs)} parents | {total_children} children embedded")
    return status


def ingest_all(data_dir: str = "data", batch_size: int = 20) -> None:
    """
    Ingest all .txt files in data_dir.
    - Unchanged files → skipped entirely
    - Changed files → old vectors deleted → re-ingested fresh (no duplicates)
    - New files → ingested fresh
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

    results = {'new': 0, 'updated': 0, 'skipped': 0}

    try:
        for file_path in txt_files:
            logger.info(f"--- Checking: {file_path} ---")
            status = ingest_file(file_path, pg_conn, qdrant_client, batch_size)
            results[status] += 1

        logger.info("=== Ingestion Complete ===")
        logger.info(f"  New files     : {results['new']}")
        logger.info(f"  Updated files : {results['updated']}")
        logger.info(f"  Skipped files : {results['skipped']}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        pg_conn.rollback()
        raise
    finally:
        pg_pool.putconn(pg_conn)
        qdrant_client.close()


def _upsert_with_retry(client: QdrantClient, points: List[PointStruct], max_retries=3):
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (attempt + 1) * 2
            logger.warning(f"Upsert failed, retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)


class ParentChildRetriever:
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


retriever = None


def get_retriever():
    global retriever
    if retriever is None:
        retriever = ParentChildRetriever(k=RETRIEVER_K)
    return retriever