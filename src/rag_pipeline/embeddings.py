"""Embedding backends. `fastembed` (ONNX, no PyTorch) is the default; `hash` is for tests and CI."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Not semantic, but stable and dependency-free."""

    name = "hash"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    embed_queries = embed


class FastEmbedder:
    """Sentence embeddings via fastembed (default model: BAAI/bge-small-en-v1.5, 384 dimensions)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        from fastembed import TextEmbedding

        self.name = model_name
        self.dim = dim
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.query_embed(texts)]


def get_embedder(backend: str, model_name: str, dim: int) -> Embedder:
    if backend == "hash":
        return HashEmbedder(dim)
    if backend == "fastembed":
        return FastEmbedder(model_name, dim)
    raise ValueError(f"unknown embedding backend: {backend!r} (use 'fastembed' or 'hash')")
