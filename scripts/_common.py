import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_pipeline import config  # noqa: E402
from rag_pipeline.embeddings import get_embedder  # noqa: E402
from rag_pipeline.store import PgVectorStore  # noqa: E402


def open_store_and_embedder():
    embedder = get_embedder(config.EMBEDDING_BACKEND, config.EMBEDDING_MODEL, config.EMBEDDING_DIM)
    return PgVectorStore(config.DSN, config.EMBEDDING_DIM), embedder
