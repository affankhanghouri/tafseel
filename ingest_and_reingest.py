"""
reset_and_reingest.py
=====================
STEP 1 — Wipes ALL data from both cloud databases:
         • Qdrant Cloud  → deletes the entire collection and recreates it fresh
         • Supabase      → truncates parent_chunks + ingested_files tables

STEP 2 — Re-ingests all .txt files from data/ folder cleanly.

Usage:
    python reset_and_reingest.py
    python reset_and_reingest.py --data-dir data
    python reset_and_reingest.py --reset-only      # only wipe, skip ingestion
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSIONS,
    SUPABASE_DB_URL,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from psycopg2.pool import SimpleConnectionPool


# ── Step 1a: Wipe Qdrant ─────────────────────────────────────────────────────

def reset_qdrant() -> None:
    logger.info("━━━ QDRANT RESET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if QDRANT_URL:
        logger.info(f"Connecting to Qdrant Cloud: {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    else:
        logger.info("Connecting to local Qdrant (path=qdrant_db)")
        client = QdrantClient(path="qdrant_db")

    # Check if collection exists
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION in existing:
        logger.info(f"  Deleting collection '{QDRANT_COLLECTION}'...")
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"  ✓ Collection deleted.")
    else:
        logger.info(f"  Collection '{QDRANT_COLLECTION}' did not exist — nothing to delete.")

    # Recreate fresh collection
    logger.info(f"  Recreating collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIMENSIONS}, cosine)...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
    )
    logger.info(f"  ✓ Fresh empty collection created.")

    info = client.get_collection(QDRANT_COLLECTION)
    points_count = info.points_count if hasattr(info, "points_count") else 0
    logger.info(f"  Collection status: {info.status} | points count: {points_count}")
    client.close()


# ── Step 1b: Wipe Supabase ────────────────────────────────────────────────────

def reset_supabase() -> None:
    logger.info("━━━ SUPABASE RESET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    pool = SimpleConnectionPool(1, 2, SUPABASE_DB_URL, sslmode='require')
    conn = pool.getconn()

    try:
        with conn.cursor() as cur:
            # Check what's in there before wiping
            cur.execute("SELECT COUNT(*) FROM parent_chunks")
            parent_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM ingested_files")
            file_count = cur.fetchone()[0]

            logger.info(f"  Before reset: {parent_count} parent chunks, {file_count} tracked files.")

            # Truncate both tables
            logger.info("  Truncating parent_chunks...")
            cur.execute("TRUNCATE TABLE parent_chunks RESTART IDENTITY CASCADE")

            logger.info("  Truncating ingested_files...")
            cur.execute("TRUNCATE TABLE ingested_files RESTART IDENTITY CASCADE")

        conn.commit()

        # Confirm zeroed out
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM parent_chunks")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT COUNT(*) FROM ingested_files")
            assert cur.fetchone()[0] == 0

        logger.info("  ✓ Both tables emptied and confirmed.")

    except Exception as e:
        conn.rollback()
        logger.error(f"  Supabase reset failed: {e}")
        raise
    finally:
        pool.putconn(conn)
        pool.closeall()


# ── Step 2: Re-ingest ─────────────────────────────────────────────────────────

def run_ingestion(data_dir: str) -> None:
    logger.info("━━━ INGESTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    from src.ingestion import ingest_all
    ingest_all(data_dir=data_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reset cloud DBs and re-ingest NADRA data")
    parser.add_argument("--data-dir",    default="data",  help="Folder containing .txt files (default: data/)")
    parser.add_argument("--reset-only",  action="store_true", help="Only wipe databases, skip ingestion")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║          NIA — FULL DATABASE RESET & REINGEST            ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── RESET ──
    reset_qdrant()
    reset_supabase()
    logger.info("━━━ RESET COMPLETE — Both databases are now empty ━━━━━━━━━")

    if args.reset_only:
        logger.info("--reset-only flag set. Skipping ingestion. Done.")
        return

    # ── REINGEST ──
    run_ingestion(args.data_dir)
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║              ALL DONE — Clean data loaded ✓              ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()