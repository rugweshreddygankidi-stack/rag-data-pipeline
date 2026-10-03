"""Retrieval evaluation on a labeled set of (query, expected doc) pairs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalQuery:
    query: str
    doc_id: str


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            rows.append(EvalQuery(obj["query"], str(obj["doc_id"])))
    return rows


def recall_at_k(ranked_doc_ids: list[list[str]], expected: list[str], k: int) -> float:
    """Fraction of queries whose expected document appears among the top-k retrieved chunks."""
    if not expected:
        return 0.0
    hits = sum(1 for got, want in zip(ranked_doc_ids, expected) if want in got[:k])
    return hits / len(expected)


def mean_reciprocal_rank(ranked_doc_ids: list[list[str]], expected: list[str]) -> float:
    if not expected:
        return 0.0
    total = 0.0
    for got, want in zip(ranked_doc_ids, expected):
        if want in got:
            total += 1.0 / (got.index(want) + 1)
    return total / len(expected)


def evaluate(queries: list[EvalQuery], embedder, store, k: int = 5) -> dict:
    vectors = embedder.embed_queries([q.query for q in queries])
    ranked = [[hit.doc_id for hit in store.search(vec, k)] for vec in vectors]
    expected = [q.doc_id for q in queries]
    return {
        "queries": len(queries),
        "k": k,
        f"recall_at_{k}": round(recall_at_k(ranked, expected, k), 4),
        "mrr": round(mean_reciprocal_rank(ranked, expected), 4),
    }
