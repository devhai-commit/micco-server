import os

# ─── Security ────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "docvault-super-secret-key-change-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ─── Database (PostgreSQL via TimescaleDB Docker) ────────────
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "micco")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ─── Embedding ───────────────────────────────────────────────
# EMBEDDING_PROVIDER: "bge" (local BAAI/bge-m3) | "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")          # used when provider=bge
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # used when provider=openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_DIMENSIONS = 512  # matches vector(512) column in DB

# ─── File Storage ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
