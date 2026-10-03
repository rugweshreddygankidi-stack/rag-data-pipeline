"""PostgreSQL + pgvector store."""
from __future__ import annotations

from datetime import datetime

from .models import ChunkRecord, SearchHit

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          text PRIMARY KEY,
    doc_hash        text NOT NULL,
    source_file     text,
    pii_redactions  integer NOT NULL DEFAULT 0,
    chunk_count     integer NOT NULL DEFAULT 0,
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      bigserial PRIMARY KEY,
    doc_id        text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index   integer NOT NULL,
    content       text NOT NULL,
    content_hash  text NOT NULL UNIQUE,
    embedding     vector({dim}) NOT NULL,
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);
"""


class PgVectorStore:
    def __init__(self, dsn: str, dim: int = 384):
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self.conn = psycopg.connect(dsn)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(SCHEMA.format(dim=dim))
        self.conn.commit()
        register_vector(self.conn)

    def close(self) -> None:
        self.conn.close()

    def get_document_hash(self, doc_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT doc_hash FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
        return row[0] if row else None

    def existing_chunk_hashes(self, hashes: list[str], exclude_doc_id: str) -> set[str]:
        if not hashes:
            return set()
        with self.conn.cursor() as cur:
            cur.execute("SELECT content_hash FROM chunks WHERE content_hash = ANY(%s) AND doc_id <> %s",
                        (hashes, exclude_doc_id))
            return {r[0] for r in cur.fetchall()}

    def replace_document(self, doc_id: str, doc_hash: str, source_file: str, pii_redactions: int,
                         chunks: list[ChunkRecord]) -> None:
        """Atomically replace all chunks of a document (idempotent; safe to re-run)."""
        import numpy as np

        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                cur.execute(
                    """INSERT INTO documents (doc_id, doc_hash, source_file, pii_redactions, chunk_count, updated_at)
                       VALUES (%s, %s, %s, %s, %s, now())
                       ON CONFLICT (doc_id) DO UPDATE SET doc_hash = EXCLUDED.doc_hash,
                         source_file = EXCLUDED.source_file, pii_redactions = EXCLUDED.pii_redactions,
                         chunk_count = EXCLUDED.chunk_count, updated_at = now()""",
                    (doc_id, doc_hash, source_file, pii_redactions, len(chunks)))
                cur.executemany(
                    """INSERT INTO chunks (doc_id, chunk_index, content, content_hash, embedding)
                       VALUES (%s, %s, %s, %s, %s) ON CONFLICT (content_hash) DO NOTHING""",
                    [(doc_id, c.chunk_index, c.content, c.content_hash, np.asarray(c.embedding, dtype=np.float32))
                     for c in chunks])

    def search(self, query_embedding: list[float], k: int) -> list[SearchHit]:
        import numpy as np

        vec = np.asarray(query_embedding, dtype=np.float32)
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT doc_id, chunk_index, content, 1 - (embedding <=> %s) AS score
                   FROM chunks ORDER BY embedding <=> %s LIMIT %s""", (vec, vec, k))
            return [SearchHit(*row) for row in cur.fetchall()]

    def last_ingested_at(self) -> datetime | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT max(ingested_at) FROM chunks")
            return cur.fetchone()[0]

    def sample_chunks(self, limit: int) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute("""SELECT doc_id, chunk_index, content, content_hash, ingested_at
                           FROM chunks ORDER BY random() LIMIT %s""", (limit,))
            cols = ["doc_id", "chunk_index", "content", "content_hash", "ingested_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def counts(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)")
            docs, chunks = cur.fetchone()
        return {"documents": docs, "chunks": chunks}
