from datetime import datetime, timedelta, timezone

import pytest

from rag_pipeline.embeddings import HashEmbedder
from rag_pipeline.evaluation import EvalQuery, evaluate, load_eval_queries, mean_reciprocal_rank, recall_at_k
from rag_pipeline.memory_store import InMemoryStore
from rag_pipeline.models import Document
from rag_pipeline.pipeline import ingest_documents
from rag_pipeline.quality import check_freshness


def test_recall_at_k_basic():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["q", "a", "b"]]
    assert recall_at_k(ranked, ["b", "a", "a"], k=2) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, ["b", "a", "a"], k=3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, ["c", "a", "a"], k=1) == 0.0


def test_recall_with_no_queries():
    assert recall_at_k([], [], 5) == 0.0 and mean_reciprocal_rank([], []) == 0.0


def test_recall_is_monotonic_in_k():
    ranked = [["a", "b", "c", "d"]] * 4
    expected = ["a", "b", "c", "d"]
    values = [recall_at_k(ranked, expected, k) for k in range(1, 5)]
    assert values == sorted(values) and values[-1] == 1.0


def test_mrr():
    assert mean_reciprocal_rank([["a", "b"], ["x", "a"], ["m", "n"]], ["a", "a", "a"]) == pytest.approx((1 + 0.5 + 0) / 3)


def test_load_eval_queries(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"query": "what?", "doc_id": 5}\n\n{"query": "why?", "doc_id": "d2"}\n')
    assert load_eval_queries(path) == [EvalQuery("what?", "5"), EvalQuery("why?", "d2")]


def test_end_to_end_retrieval_on_small_corpus():
    topics = {"rates": "central bank interest rates inflation", "storm": "coastal storm flooding rain harbour",
              "chip": "semiconductor factory advanced chip production", "vaccine": "seasonal vaccine campaign older adults"}
    emb, store = HashEmbedder(256), InMemoryStore()
    docs = [Document(t, (s + " ") * 3 + " ".join(f"{t}filler{i}" for i in range(80))) for t, s in topics.items()]
    ingest_documents(docs, store, emb, max_words=60, overlap=5)
    queries = [EvalQuery(s, t) for t, s in topics.items()]
    result = evaluate(queries, emb, store, k=3)
    assert result["recall_at_3"] == 1.0 and result["queries"] == 4 and 0 < result["mrr"] <= 1


NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def test_freshness_ok_when_ingest_is_newer_than_source():
    assert check_freshness(NOW - timedelta(hours=1), NOW, timedelta(hours=2)) == (True, 0.0)


def test_freshness_ok_within_allowed_lag():
    ok, lag = check_freshness(NOW, NOW - timedelta(minutes=90), timedelta(hours=2))
    assert ok and lag == 5400.0


def test_freshness_stale_beyond_allowed_lag():
    ok, lag = check_freshness(NOW, NOW - timedelta(hours=5), timedelta(hours=2))
    assert not ok and lag == 18000.0


def test_freshness_when_nothing_ingested_yet():
    assert check_freshness(NOW, None, timedelta(hours=2))[0] is False


def test_freshness_without_source_files_is_fine():
    assert check_freshness(None, None, timedelta(hours=2)) == (True, 0.0)


def make_rows(**override):
    row = {"doc_id": "d", "chunk_index": 0, "content": "a perfectly reasonable chunk of text here",
           "content_hash": "h1", "ingested_at": NOW}
    row.update(override)
    return row


def test_great_expectations_passes_clean_chunks():
    pytest.importorskip("great_expectations")
    from rag_pipeline.quality import run_chunk_checks

    report = run_chunk_checks([make_rows(), make_rows(content_hash="h2", content="another valid chunk of reasonable length")])
    assert report.success, report.failures


def test_great_expectations_flags_leaked_pii_duplicates_and_short_chunks():
    pytest.importorskip("great_expectations")
    from rag_pipeline.quality import run_chunk_checks

    rows = [make_rows(content="contact jane@example.com for the full text of this chunk"),
            make_rows(content="short", content_hash="h1")]
    report = run_chunk_checks(rows)
    assert not report.success and len(report.failures) >= 2


def test_quality_report_for_no_rows():
    pytest.importorskip("great_expectations")
    from rag_pipeline.quality import run_chunk_checks

    assert run_chunk_checks([]).success is False
