"""Settings read from environment variables (defaults suit the docker-compose setup)."""
import os

DSN = os.environ.get("RAG_DSN", "postgresql://rag:rag@localhost:5432/rag")
DATA_DIR = os.environ.get("DATA_DIR", "data")
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "fastembed")  # fastembed | hash
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
CHUNK_MAX_WORDS = int(os.environ.get("CHUNK_MAX_WORDS", "180"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "30"))
MAX_SOURCE_LAG_MINUTES = int(os.environ.get("MAX_SOURCE_LAG_MINUTES", "120"))
MIN_RECALL_AT_5 = float(os.environ.get("MIN_RECALL_AT_5", "0.0"))  # 0 = log only, never fail
