"""In-memory VectorStore with the same interface as PgVectorStore (used by tests and quick demos)."""
from __future__ import annotations

from datetime import datetime, timezone

from .models import ChunkRecord, SearchHit


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryStore:
    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}  # content_hash -> row (mirrors UNIQUE(content_hash))

    def get_document_hash(self, doc_id: str) -> str | None:
        doc = self.documents.get(doc_id)
        return doc["doc_hash"] if doc else None

    def existing_chunk_hashes(self, hashes: list[str], exclude_doc_id: str) -> set[str]:
        return {h for h in hashes if h in self.chunks and self.chunks[h]["doc_id"] != exclude_doc_id}

    def replace_document(self, doc_id: str, doc_hash: str, source_file: str, pii_redactions: int,
                         chunks: list[ChunkRecord]) -> None:
        now = datetime.now(timezone.utc)
        self.chunks = {h: row for h, row in self.chunks.items() if row["doc_id"] != doc_id}
        for chunk in chunks:
            self.chunks.setdefault(chunk.content_hash, {
                "doc_id": doc_id, "chunk_index": chunk.chunk_index, "content": chunk.content,
                "content_hash": chunk.content_hash, "embedding": chunk.embedding, "ingested_at": now})
        self.documents[doc_id] = {"doc_hash": doc_hash, "source_file": source_file,
                                  "pii_redactions": pii_redactions, "updated_at": now}

    def search(self, query_embedding: list[float], k: int) -> list[SearchHit]:
        scored = sorted(self.chunks.values(), key=lambda r: _dot(query_embedding, r["embedding"]), reverse=True)
        return [SearchHit(r["doc_id"], r["chunk_index"], r["content"], _dot(query_embedding, r["embedding"]))
                for r in scored[:k]]

    def last_ingested_at(self) -> datetime | None:
        times = [r["ingested_at"] for r in self.chunks.values()]
        return max(times) if times else None

    def sample_chunks(self, limit: int) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "embedding"} for row in list(self.chunks.values())[:limit]]

    def counts(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}
