"""
ingestion.py — Cloud Edition (Supabase + Qdrant Cloud)
======================================================
"""

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
        # CRITICAL FIX: timeout=60 for cloud operations (default is 5s, too short)
        return QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            timeout=60  # Increased from default 5.0
        )
    
    logger.info("Using local Qdrant")
    return QdrantClient(path="qdrant_db")


def _ensure_collection(client: QdrantClient) -> None:
    """Creates collection with proper error handling"""
    try:
        client.get_collection(QDRANT_COLLECTION)
        logger.info(f"Collection '{QDRANT_COLLECTION}' exists")
    except Exception as e:
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
            1, 5,  # min, max connections
            SUPABASE_DB_URL, 
            sslmode='require'
        )
    return _pg_pool


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


def ingest(source_path: str = "nadra_info.txt", batch_size: int = 20) -> None:  # Reduced from 100 to 20
    """
    Ingest documents to Cloud Qdrant + Supabase.
    Smaller batch_size (20) prevents cloud timeouts.
    """
    logger.info("=== Starting Cloud Ingestion ===")

    # Load & split
    raw_docs = TextLoader(source_path, encoding="utf-8").load()
    logger.info(f"Loaded {len(raw_docs)} documents")

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
    logger.info(f"Created {len(parent_docs)} parent chunks")

    # Connect to cloud services
    pg_pool = _get_pg_pool()
    pg_conn = pg_pool.getconn()
    
    # Initialize table
    with pg_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                metadata JSONB NOT NULL DEFAULT '{}'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parent_chunks_id ON parent_chunks(id)")
    pg_conn.commit()

    qdrant_client = _make_qdrant_client()
    _ensure_collection(qdrant_client)

    pending_points: List[PointStruct] = []
    total_children = 0

    try:
        for i, parent_doc in enumerate(parent_docs):
            parent_id = str(uuid.uuid4())
            source = parent_doc.metadata.get("source", source_path)

            # Store parent in Supabase
            _store_parent(pg_conn, parent_id, parent_doc.page_content, source, parent_doc.metadata)
            
            if i % 10 == 0:
                pg_conn.commit()  # Periodic commit
                logger.info(f"Processed {i}/{len(parent_docs)} parents...")

            # Create children
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

                # CRITICAL FIX: Smaller batches + retry logic for cloud
                if len(pending_points) >= batch_size:
                    _upsert_with_retry(qdrant_client, pending_points)
                    logger.info(f"Upserted {total_children} children total...")
                    pending_points.clear()

        # Flush remaining
        if pending_points:
            _upsert_with_retry(qdrant_client, pending_points)

        pg_conn.commit()
        logger.info(f"=== Complete: {len(parent_docs)} parents | {total_children} children ===")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
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
                wait=True  # Ensure it's written before continuing
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
        """Lazy connection initialization"""
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

        # Embed query
        query_vector = embeddings.embed_query(query)

        # Search Qdrant
        hits = self._qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=self.k,
            with_payload=True,
        ).points

        if not hits:
            logger.warning(f"No Qdrant results for: {query[:80]}...")
            return []

        # Fetch from Supabase
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


# LAZY INITIALIZATION: Don't create retriever on import
retriever = None

def get_retriever():
    """Get or create retriever singleton"""
    global retriever
    if retriever is None:
        retriever = ParentChildRetriever(k=RETRIEVER_K)
    return retriever