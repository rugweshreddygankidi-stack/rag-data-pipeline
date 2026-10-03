"""Regex-based PII redaction (emails, SSNs, phone numbers, credit cards)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
SSN_REGEX = r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
CARD_REGEX = r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
PHONE_REGEX = r"(?<![\w-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"

_EMAIL, _SSN, _CARD, _PHONE = (re.compile(p) for p in (EMAIL_REGEX, SSN_REGEX, CARD_REGEX, PHONE_REGEX))


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> RedactionResult:
    """Replace PII with [EMAIL] / [SSN] / [CARD] / [PHONE] tokens and count what was removed."""
    counts: dict[str, int] = {}

    def sub(pattern, token, value, validator=None):
        def repl(match):
            if validator and not validator(match.group(0)):
                return match.group(0)
            counts[token] = counts.get(token, 0) + 1
            return f"[{token}]"
        return pattern.sub(repl, value)

    # Order matters: long digit runs (cards) and SSNs must be handled before the phone pattern.
    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_CARD, "CARD", text, luhn_valid)
    text = sub(_SSN, "SSN", text)
    text = sub(_PHONE, "PHONE", text)
    return RedactionResult(text=text, counts=counts)
