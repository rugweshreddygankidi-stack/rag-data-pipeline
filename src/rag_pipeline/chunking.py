"""Word-window chunking with overlap."""
from __future__ import annotations


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    """Split text into chunks of at most `max_words` words; consecutive chunks share `overlap` words."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and smaller than max_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    step = max_words - overlap
    chunks: list[str] = []
    start = 0
    while True:
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            return chunks
        start += step
