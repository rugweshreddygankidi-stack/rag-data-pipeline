"""Ask the vector store a question.

    python scripts/search_demo.py "what did the minister say about interest rates?"
"""
import sys

from _common import open_store_and_embedder

store, embedder = open_store_and_embedder()
query = " ".join(sys.argv[1:]) or "central bank interest rate decision"
for hit in store.search(embedder.embed_queries([query])[0], 5):
    print(f"[{hit.score:.3f}] doc={hit.doc_id} chunk={hit.chunk_index}\n    {hit.content[:220]}...\n")
