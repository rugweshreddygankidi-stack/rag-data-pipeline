"""Data-quality gates: Great Expectations checks on stored chunks + a source-freshness check."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .redact import CARD_REGEX, EMAIL_REGEX, SSN_REGEX


@dataclass
class QualityReport:
    success: bool
    checked_rows: int
    failures: list[str] = field(default_factory=list)


def run_chunk_checks(rows: list[dict], min_chars: int = 20, max_chars: int = 6000) -> QualityReport:
    """Validate a sample of stored chunks with Great Expectations (0.18.x API)."""
    import great_expectations as gx
    import pandas as pd

    if not rows:
        return QualityReport(False, 0, ["no chunks found to validate"])

    df = gx.from_pandas(pd.DataFrame(rows))
    df.expect_column_values_to_not_be_null("doc_id")
    df.expect_column_values_to_not_be_null("content")
    df.expect_column_values_to_not_be_null("ingested_at")
    df.expect_column_value_lengths_to_be_between("content", min_value=min_chars, max_value=max_chars)
    df.expect_column_values_to_be_unique("content_hash")
    # PII must not survive redaction
    df.expect_column_values_to_not_match_regex("content", EMAIL_REGEX)
    df.expect_column_values_to_not_match_regex("content", SSN_REGEX)
    df.expect_column_values_to_not_match_regex("content", CARD_REGEX)

    result = df.validate()
    failures = [r.expectation_config.expectation_type + " " + str(r.expectation_config.kwargs.get("column", ""))
                for r in result.results if not r.success]
    return QualityReport(bool(result.success), len(rows), failures)


def newest_source_mtime(directory: str | Path) -> datetime | None:
    paths = list(Path(directory).glob("*.jsonl")) + list(Path(directory).glob("*.txt"))
    if not paths:
        return None
    return datetime.fromtimestamp(max(p.stat().st_mtime for p in paths), tz=timezone.utc)


def check_freshness(newest_source: datetime | None, last_ingested: datetime | None,
                    max_lag: timedelta) -> tuple[bool, float]:
    """The store is stale if source data is newer than the last ingest by more than `max_lag`.

    Returns (is_fresh, lag_seconds). No source data means nothing to be stale against.
    """
    if newest_source is None:
        return True, 0.0
    if last_ingested is None:
        return False, float("inf")
    lag = (newest_source - last_ingested).total_seconds()
    return lag <= max_lag.total_seconds(), max(lag, 0.0)
