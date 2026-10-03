"""Run one ingestion pass (the same code the Airflow DAG runs).

    python scripts/run_ingest.py
"""
import json

from _common import config, open_store_and_embedder
from rag_pipeline.pipeline import ingest_directory

store, embedder = open_store_and_embedder()
stats = ingest_directory(f"{config.DATA_DIR}/inbox", store, embedder,
                         max_words=config.CHUNK_MAX_WORDS, overlap=config.CHUNK_OVERLAP_WORDS)
print(json.dumps({**stats.as_dict(), **store.counts()}, indent=2))
