"""Download a public news corpus and write it as 'hourly drop' shards plus a labeled eval set.

Source: the CNN/DailyMail summarization dataset from Hugging Face. Each article's human-written
highlights become the search query, and the article itself is the expected result, which gives a
labeled retrieval set with no manual labeling. Check the dataset's license on its Hugging Face page.

    pip install datasets
    python scripts/fetch_corpus.py --docs 25000 --queries 500
"""
import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--docs", type=int, default=25_000)
parser.add_argument("--shard-size", type=int, default=1000)
parser.add_argument("--queries", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="data")
args = parser.parse_args()

from datasets import load_dataset  # noqa: E402

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"train[:{args.docs}]")
inbox, evaldir = Path(args.out, "inbox"), Path(args.out, "eval")
inbox.mkdir(parents=True, exist_ok=True)
evaldir.mkdir(parents=True, exist_ok=True)

rows = [{"doc_id": row["id"], "text": row["article"]} for row in dataset]
for shard_no, start in enumerate(range(0, len(rows), args.shard_size)):
    path = inbox / f"news_{shard_no:03d}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows[start:start + args.shard_size]), encoding="utf-8")

rng = random.Random(args.seed)
sample = rng.sample(range(len(dataset)), min(args.queries, len(dataset)))
with (evaldir / "queries.jsonl").open("w", encoding="utf-8") as fh:
    for i in sample:
        fh.write(json.dumps({"query": dataset[i]["highlights"].replace("\n", " "), "doc_id": dataset[i]["id"]}) + "\n")

print(f"wrote {len(rows)} documents in {-(-len(rows) // args.shard_size)} shards and {len(sample)} eval queries")
