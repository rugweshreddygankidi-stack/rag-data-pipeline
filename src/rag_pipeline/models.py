from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    text: str
    source_file: str = ""


@dataclass
class ChunkRecord:
    chunk_index: int
    content: str
    content_hash: str
    embedding: list[float]


@dataclass
class SearchHit:
    doc_id: str
    chunk_index: int
    content: str
    score: float
