"""Content hashing used for change detection and duplicate removal."""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def content_hash(text: str) -> str:
    """Hash of the normalized text: whitespace and case differences do not create 'new' content."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def document_hash(raw_text: str) -> str:
    """Hash of the raw document text; a changed hash triggers re-processing of that document."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
