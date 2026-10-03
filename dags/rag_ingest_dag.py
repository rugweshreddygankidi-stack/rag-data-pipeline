"""Hourly incremental ingestion + data-quality gates, and a daily retrieval-quality check."""
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

from rag_pipeline import config
from rag_pipeline.embeddings import get_embedder
from rag_pipeline.store import PgVectorStore

DEFAULT_ARGS = {"retries": 2, "retry_delay": timedelta(minutes=2)}


def _open():
    embedder = get_embedder(config.EMBEDDING_BACKEND, config.EMBEDDING_MODEL, config.EMBEDDING_DIM)
    return PgVectorStore(config.DSN, config.EMBEDDING_DIM), embedder


@dag(dag_id="rag_ingest_hourly", schedule="@hourly", start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
     catchup=False, max_active_runs=1, default_args=DEFAULT_ARGS, tags=["rag", "ingest"])
def rag_ingest_hourly():
    @task
    def ingest() -> dict:
        from rag_pipeline.pipeline import ingest_directory

        store, embedder = _open()
        try:
            stats = ingest_directory(f"{config.DATA_DIR}/inbox", store, embedder,
                                     max_words=config.CHUNK_MAX_WORDS, overlap=config.CHUNK_OVERLAP_WORDS)
            return {**stats.as_dict(), **store.counts()}
        finally:
            store.close()

    @task
    def quality_checks(_ingest_stats: dict) -> dict:
        from rag_pipeline.quality import run_chunk_checks

        store, _ = _open()
        try:
            report = run_chunk_checks(store.sample_chunks(2000))
        finally:
            store.close()
        if not report.success:
            raise AirflowFailException(f"chunk quality checks failed: {report.failures}")
        return {"checked_rows": report.checked_rows}

    @task
    def freshness_check(_quality: dict) -> dict:
        from rag_pipeline.quality import check_freshness, newest_source_mtime

        store, _ = _open()
        try:
            fresh, lag = check_freshness(newest_source_mtime(f"{config.DATA_DIR}/inbox"),
                                         store.last_ingested_at(),
                                         timedelta(minutes=config.MAX_SOURCE_LAG_MINUTES))
        finally:
            store.close()
        if not fresh:
            raise AirflowFailException(f"vector store is stale: {lag:.0f}s behind the newest source file")
        return {"source_lag_seconds": lag}

    freshness_check(quality_checks(ingest()))


@dag(dag_id="rag_eval_daily", schedule="@daily", start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
     catchup=False, max_active_runs=1, default_args=DEFAULT_ARGS, tags=["rag", "evaluation"])
def rag_eval_daily():
    @task
    def retrieval_quality() -> dict:
        from rag_pipeline.evaluation import evaluate, load_eval_queries

        store, embedder = _open()
        try:
            result = evaluate(load_eval_queries(f"{config.DATA_DIR}/eval/queries.jsonl"), embedder, store, k=5)
        finally:
            store.close()
        if result["recall_at_5"] < config.MIN_RECALL_AT_5:
            raise AirflowFailException(f"recall@5 {result['recall_at_5']} below {config.MIN_RECALL_AT_5}")
        return result

    retrieval_quality()


rag_ingest_hourly()
rag_eval_daily()
