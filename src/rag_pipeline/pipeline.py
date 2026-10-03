"""Ingestion: read documents -> skip unchanged -> redact PII -> chunk -> dedupe -> embed -> store."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from .chunking import chunk_text
from .dedupe import content_hash, document_hash
from .models import ChunkRecord, Document
from .redact import redact


@dataclass
class IngestStats:
    files: int = 0
    documents_seen: int = 0
    documents_unchanged: int = 0
    documents_empty: int = 0
    documents_ingested: int = 0
    chunks_created: int = 0
    chunks_deduplicated: int = 0
    pii_redactions: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Pending:
    doc: Document
    doc_hash: str
    pii: int
    items: list[tuple[int, str, str]] = field(default_factory=list)  # (chunk_index, text, hash)


def iter_documents(directory: str | Path) -> Iterator[Document]:
    """Yield documents from *.jsonl ({"doc_id", "text"} per line) and *.txt (doc_id = file name)."""
    root = Path(directory)
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    yield Document(str(row["doc_id"]), str(row.get("text", "")), path.name)
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"{path.name}:{line_no}: bad document line ({exc})") from exc
    for path in sorted(root.glob("*.txt")):
        yield Document(path.stem, path.read_text(encoding="utf-8"), path.name)


def ingest_documents(documents, store, embedder, *, max_words: int = 180, overlap: int = 30,
                     flush_chunks: int = 256) -> IngestStats:
    """Idempotent, incremental ingestion. Re-running on unchanged input writes nothing."""
    stats = IngestStats()
    started = time.monotonic()
    pending: list[_Pending] = []
    buffered_hashes: set[str] = set()
    buffered_chunks = 0

    def flush() -> None:
        nonlocal pending, buffered_hashes, buffered_chunks
        texts = [text for p in pending for _, text, _ in p.items]
        vectors = embedder.embed(texts) if texts else []
        position = 0
        for p in pending:
            records = []
            for chunk_index, text, digest in p.items:
                records.append(ChunkRecord(chunk_index, text, digest, vectors[position]))
                position += 1
            store.replace_document(p.doc.doc_id, p.doc_hash, p.doc.source_file, p.pii, records)
            stats.documents_ingested += 1
            stats.chunks_created += len(records)
        pending, buffered_hashes, buffered_chunks = [], set(), 0

    for doc in documents:
        stats.documents_seen += 1
        doc_hash = document_hash(doc.text)
        if store.get_document_hash(doc.doc_id) == doc_hash:
            stats.documents_unchanged += 1
            continue
        if not doc.text.strip():
            stats.documents_empty += 1
            continue

        redacted = redact(doc.text)
        stats.pii_redactions += redacted.total
        chunks = chunk_text(redacted.text, max_words, overlap)
        hashes = [content_hash(c) for c in chunks]
        already_stored = store.existing_chunk_hashes(hashes, exclude_doc_id=doc.doc_id)

        item = _Pending(doc, doc_hash, redacted.total)
        seen_here: set[str] = set()
        for index, (text, digest) in enumerate(zip(chunks, hashes)):
            if digest in already_stored or digest in buffered_hashes or digest in seen_here:
                stats.chunks_deduplicated += 1
                continue
            seen_here.add(digest)
            item.items.append((index, text, digest))
        buffered_hashes |= seen_here
        buffered_chunks += len(item.items)
        pending.append(item)
        if buffered_chunks >= flush_chunks:
            flush()

    flush()
    stats.seconds = round(time.monotonic() - started, 2)
    return stats


def ingest_directory(directory, store, embedder, **kwargs) -> IngestStats:
    stats = ingest_documents(iter_documents(directory), store, embedder, **kwargs)
    root = Path(directory)
    stats.files = len(list(root.glob("*.jsonl"))) + len(list(root.glob("*.txt")))
    return stats
