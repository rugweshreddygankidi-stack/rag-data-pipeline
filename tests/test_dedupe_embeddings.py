import math

import pytest

from rag_pipeline.dedupe import content_hash, document_hash, normalize
from rag_pipeline.embeddings import HashEmbedder, get_embedder


def test_content_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World\n") == content_hash("hello world")


def test_content_hash_differs_for_different_text():
    assert content_hash("hello world") != content_hash("hello there")


def test_document_hash_is_sensitive_to_whitespace():
    assert document_hash("a b") != document_hash("a  b")


def test_normalize():
    assert normalize("  A  B\tC ") == "a b c"


def test_hash_embedder_dimension_and_unit_norm():
    vec = HashEmbedder(64).embed(["central bank interest rates"])[0]
    assert len(vec) == 64 and math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)


def test_hash_embedder_is_deterministic():
    e = HashEmbedder(64)
    assert e.embed(["same text"]) == e.embed(["same text"])


def test_hash_embedder_empty_text_is_zero_vector():
    assert set(HashEmbedder(16).embed([""])[0]) == {0.0}


def test_similar_texts_score_higher_than_unrelated():
    e = HashEmbedder(256)
    a, b, c = e.embed(["interest rates inflation central bank", "central bank raises interest rates",
                       "football match score goal"])
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_embed_queries_matches_embed_for_hash_backend():
    e = HashEmbedder(32)
    assert e.embed_queries(["x y"]) == e.embed(["x y"])


@pytest.mark.parametrize("backend", ["hash"])
def test_get_embedder_hash(backend):
    assert get_embedder(backend, "ignored", 48).dim == 48


def test_get_embedder_unknown_backend():
    with pytest.raises(ValueError):
        get_embedder("nope", "m", 8)
