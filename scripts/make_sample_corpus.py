"""Create a small synthetic corpus (no downloads) with PII, exact duplicates and a document update.

Useful for a 2-minute demo of redaction, deduplication and incremental ingestion.

    python scripts/make_sample_corpus.py
"""
import json
from pathlib import Path

TOPICS = {
    "rates": "The central bank held interest rates steady and said inflation is easing gradually.",
    "storm": "A coastal storm brought heavy rain and flooding to several harbour towns overnight.",
    "chip": "A semiconductor maker announced a new factory to expand advanced chip production.",
    "vaccine": "Health officials reported that the seasonal vaccine campaign reached most older adults.",
    "election": "Voters lined up early as turnout in the regional election exceeded forecasts.",
}
inbox = Path("data/inbox")
inbox.mkdir(parents=True, exist_ok=True)

docs = []
for i, (topic, sentence) in enumerate(TOPICS.items()):
    body = " ".join([sentence] + [f"Analysts discussed the {topic} story in detail, point {n}." for n in range(60)])
    if i == 0:
        body += " Contact the press office at press@example.com or 555-123-4567."
    docs.append({"doc_id": f"sample-{topic}", "text": body})
docs.append({"doc_id": "sample-rates-copy", "text": docs[0]["text"]})  # exact duplicate content

inbox.joinpath("sample_000.jsonl").write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
Path("data/eval").mkdir(parents=True, exist_ok=True)
Path("data/eval/queries.jsonl").write_text(
    "\n".join(json.dumps({"query": s, "doc_id": f"sample-{t}"}) for t, s in TOPICS.items()), encoding="utf-8")
print(f"wrote {len(docs)} sample documents and {len(TOPICS)} eval queries")
