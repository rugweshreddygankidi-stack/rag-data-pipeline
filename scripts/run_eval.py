"""Measure retrieval quality (recall@k, MRR) on the labeled query set.

    python scripts/run_eval.py --k 5
"""
import argparse
import json

from _common import config, open_store_and_embedder
from rag_pipeline.evaluation import evaluate, load_eval_queries

parser = argparse.ArgumentParser()
parser.add_argument("--queries", default=f"{config.DATA_DIR}/eval/queries.jsonl")
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

store, embedder = open_store_and_embedder()
print(json.dumps(evaluate(load_eval_queries(args.queries), embedder, store, args.k), indent=2))
