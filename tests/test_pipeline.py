import json

import pytest

from rag_pipeline.embeddings import HashEmbedder
from rag_pipeline.memory_store import InMemoryStore
from rag_pipeline.models import Document
from rag_pipeline.pipeline import ingest_directory, ingest_documents, iter_documents

EMB = HashEmbedder(64)


def long_text(topic, n=300):
    return " ".join(f"{topic}{i}" for i in range(n))


def run(docs, store=None, **kw):
    store = store or InMemoryStore()
    kw.setdefault("max_words", 100)
    kw.setdefault("overlap", 10)
    return ingest_documents(docs, store, EMB, **kw), store


def test_ingest_new_documents():
    stats, store = run([Document("a", long_text("alpha")), Document("b", long_text("beta"))])
    assert stats.documents_ingested == 2 and stats.documents_seen == 2
    assert store.counts()["documents"] == 2 and store.counts()["chunks"] == stats.chunks_created > 2


def test_second_run_on_unchanged_input_writes_nothing():
    docs = [Document("a", long_text("alpha"))]
    _, store = run(docs)
    stats, _ = run(docs, store)
    assert stats.documents_unchanged == 1 and stats.documents_ingested == 0 and stats.chunks_created == 0


def test_changed_document_replaces_its_chunks():
    _, store = run([Document("a", long_text("alpha"))])
    old_hashes = set(store.chunks)
    stats, _ = run([Document("a", long_text("gamma"))], store)
    assert stats.documents_ingested == 1
    assert not (old_hashes & set(store.chunks))      # old chunks gone
    assert store.counts()["documents"] == 1


def test_only_changed_documents_are_reprocessed():
    _, store = run([Document("a", long_text("alpha")), Document("b", long_text("beta"))])
    stats, _ = run([Document("a", long_text("alpha")), Document("b", long_text("beta2"))], store)
    assert stats.documents_unchanged == 1 and stats.documents_ingested == 1


def test_duplicate_content_across_documents_is_stored_once():
    stats, store = run([Document("a", long_text("alpha")), Document("copy", long_text("alpha"))])
    assert stats.chunks_deduplicated > 0
    assert {row["doc_id"] for row in store.chunks.values()} == {"a"}
    assert "copy" in store.documents          # remembered, so it is not reprocessed every hour


def test_duplicate_documents_in_same_flush_window_are_deduped():
    stats, store = run([Document("a", long_text("x")), Document("b", long_text("x"))], flush_chunks=10_000)
    assert len(store.chunks) == stats.chunks_created


def test_repeated_chunk_within_one_document_is_deduped():
    block = long_text("rep", 100)
    stats, store = run([Document("a", f"{block} {block}")], max_words=100, overlap=0)
    assert stats.chunks_deduplicated >= 1 and len(store.chunks) == 1


def test_pii_is_redacted_before_storage():
    stats, store = run([Document("a", "Email me at jane@example.com or call 555-123-4567. " + long_text("p", 50))])
    stored = " ".join(row["content"] for row in store.chunks.values())
    assert "jane@example.com" not in stored and "555-123-4567" not in stored
    assert "[EMAIL]" in stored and stats.pii_redactions == 2


def test_empty_documents_are_skipped_and_counted():
    stats, store = run([Document("blank", "  \n "), Document("ok", long_text("ok", 20))])
    assert stats.documents_empty == 1 and stats.documents_ingested == 1
    assert "blank" not in store.documents


def test_small_flush_size_gives_same_result_as_large():
    docs = [Document(f"d{i}", long_text(f"t{i}", 250)) for i in range(8)]
    _, small = run(docs, flush_chunks=1)
    _, large = run(docs, flush_chunks=10_000)
    assert set(small.chunks) == set(large.chunks)
    assert small.counts() == large.counts()


def test_chunk_index_is_position_in_original_document():
    block = long_text("rep", 100)
    _, store = run([Document("a", f"{block} {long_text('uniq', 100)}")], max_words=100, overlap=0)
    assert sorted(r["chunk_index"] for r in store.chunks.values()) == [0, 1]


def test_stored_embeddings_have_embedder_dimension():
    _, store = run([Document("a", long_text("alpha", 50))])
    assert all(len(r["embedding"]) == EMB.dim for r in store.chunks.values())


def test_iter_documents_reads_jsonl_and_txt(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"doc_id": "j1", "text": "hello"}) + "\n\n" +
                                      json.dumps({"doc_id": 7, "text": "world"}))
    (tmp_path / "note.txt").write_text("plain text")
    docs = {d.doc_id: d for d in iter_documents(tmp_path)}
    assert set(docs) == {"j1", "7", "note"} and docs["note"].source_file == "note.txt"


def test_iter_documents_reports_bad_line(tmp_path):
    (tmp_path / "bad.jsonl").write_text('{"doc_id": "x", "text": "ok"}\nnot json\n')
    with pytest.raises(ValueError) as exc:
        list(iter_documents(tmp_path))
    assert "bad.jsonl:2" in str(exc.value)


def test_ingest_directory_counts_files(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"doc_id": "1", "text": long_text("z", 40)}))
    (tmp_path / "b.txt").write_text(long_text("y", 40))
    stats = ingest_directory(tmp_path, InMemoryStore(), EMB, max_words=100, overlap=10)
    assert stats.files == 2 and stats.documents_ingested == 2
