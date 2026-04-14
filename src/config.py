import os
from dotenv import load_dotenv

load_dotenv()

# ── Vector Store (Qdrant) ─────────────────────────────────────────────────────
# For local development:  set QDRANT_URL to None  → uses on-disk file storage
# For production server:  set QDRANT_URL=http://localhost:6333 (or cloud URL)
# For Qdrant Cloud:       set QDRANT_URL + QDRANT_API_KEY in your .env
QDRANT_URL        = os.getenv("QDRANT_URL", None)       # None = local file mode
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", None)   # only needed for Qdrant Cloud
QDRANT_PATH       = "qdrant_db"                          # local on-disk path (ignored if QDRANT_URL set)
QDRANT_COLLECTION = "nia_child_chunks"

# ── Parent Chunk Store (SQLite) ───────────────────────────────────────────────
SQLITE_PATH = "parent_chunks.db"

# ── Chunking Strategy ─────────────────────────────────────────────────────────
# Parent: large, context-rich blocks stored in SQLite
# Child:  small, precise blocks stored in Qdrant with a pointer to their parent
PARENT_CHUNK_SIZE    = 1000   # characters (≈ 200-250 tokens)
PARENT_CHUNK_OVERLAP = 200    # overlap keeps boundary sentences intact
CHILD_CHUNK_SIZE     = 300    # characters (≈ 60-80 tokens) — tight, precise
CHILD_CHUNK_OVERLAP  = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
# How many child chunks to pull from Qdrant per query.
# After deduplication by parent_id, the LLM may receive fewer but larger chunks.
RETRIEVER_K = 5   # fetch 5 child chunks → dedup → typically 2-4 unique parents

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536   # fixed for text-embedding-3-small

# ── LangGraph Self-RAG Loop Limits ───────────────────────────────────────────
MAX_RETRIES       = 3   # max revise-answer passes in the IsSUP loop
MAX_REWRITE_TRIES = 3   # max rewrite-question passes in the IsUSE loop