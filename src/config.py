import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "production")

# ── Supabase PostgreSQL (Replaces SQLite) ────────────────────────────────────
# Format: postgresql://postgres:[DB-PASSWORD]@db.fpnzkzkzwjwrpigmupnl.supabase.co:5432/postgres
# NOTE: You need your Database password from Supabase Dashboard → Database → Connection String
SUPABASE_DB_URL = os.getenv('supabase_db_url')
    

# ── Qdrant Cloud (Replaces Local File Storage) ───────────────────────────────
# Using your provided cloud instance
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nia_child_chunks")

# ── Chunking (Unchanged) ────────────────────────────────────────────────────
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50

# ── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVER_K = 5

# ── Embedding ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# ── LangGraph Limits ────────────────────────────────────────────────────────
MAX_RETRIES = 3
MAX_REWRITE_TRIES = 3

# Local dev fallback (when ENV=development)
if ENV == "development":
    QDRANT_URL = os.getenv("QDRANT_URL", None)  # None = local file mode
    QDRANT_API_KEY = None
    SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///./parent_chunks.db")