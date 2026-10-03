# Estimated results

> **Estimated values.** Derived from service limits, the pipeline configuration and typical laptop performance; actual figures vary by machine and environment.

| Metric | Estimated | How the estimate was derived |
|---|---|---|
| Documents ingested | 25,000 | `fetch_corpus.py --docs 25000` |
| Chunks stored | ~110,000-130,000 | Articles average roughly 700 words; 150-word step gives ~5 chunks each |
| First full ingest | ~10-25 min | bge-small on CPU embeds roughly 100-300 chunks/s depending on the machine |
| Incremental run, nothing changed | ~1-3 min | Hashing plus one lookup per document; no embedding |
| Duplicate chunks removed | under 1% | News articles rarely repeat whole 180-word passages |
| PII items redacted | a few hundred | Phone numbers and emails appear in some articles |
| Recall@5 | ~0.80-0.90 | Highlights are summaries of their own article, so a small embedding model usually finds the source in the top 5 |
| MRR | ~0.65-0.80 | Same reasoning; the source is often but not always ranked first |
| Unit tests | 74 test cases (+3 Great Expectations tests in CI) | Counted locally including parametrized cases |

## Experiments to record
1. `EMBEDDING_BACKEND=hash` vs `fastembed`: record the recall gap.
2. `CHUNK_MAX_WORDS` 100 / 180 / 300: record recall@5 for each.
3. Edit one document and confirm only that document is re-processed.
