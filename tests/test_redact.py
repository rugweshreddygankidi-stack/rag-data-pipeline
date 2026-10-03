import pytest

from rag_pipeline.redact import luhn_valid, redact


@pytest.mark.parametrize("text", ["write to jane.doe@example.com now", "a+tag@sub.domain.co.uk", "X_Y-z@a-b.org"])
def test_emails_redacted(text):
    result = redact(text)
    assert "[EMAIL]" in result.text and "@" not in result.text
    assert result.counts == {"EMAIL": 1}


@pytest.mark.parametrize("phone", ["555-123-4567", "(555) 123-4567", "555.123.4567", "+1 555 123 4567", "5551234567"])
def test_phone_formats_redacted(phone):
    result = redact(f"call {phone} today")
    assert result.text == "call [PHONE] today"


def test_ssn_redacted_and_not_counted_as_phone():
    result = redact("SSN 123-45-6789 on file")
    assert result.text == "SSN [SSN] on file"
    assert result.counts == {"SSN": 1}


@pytest.mark.parametrize("card", ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111"])
def test_valid_card_numbers_redacted(card):
    result = redact(f"card {card} expires soon")
    assert result.counts == {"CARD": 1}
    assert "4111" not in result.text


def test_invalid_luhn_number_left_alone():
    assert redact("order 1234 5678 9012 3456 shipped").counts == {}


@pytest.mark.parametrize("text", ["In 2026 revenue grew 12.5%", "Version 3.11.4 released", "Route 66 is long", ""])
def test_non_pii_untouched(text):
    result = redact(text)
    assert result.text == text and result.total == 0


def test_multiple_pii_counts():
    result = redact("a@x.com, b@y.org, 555-123-4567 and 123-45-6789")
    assert result.counts == {"EMAIL": 2, "PHONE": 1, "SSN": 1} and result.total == 4


def test_redaction_is_idempotent():
    once = redact("mail a@x.com or 555-123-4567").text
    assert redact(once).text == once


@pytest.mark.parametrize("number,valid", [("4111111111111111", True), ("4111111111111112", False),
                                          ("79927398713", False), ("5500 0000 0000 0004", True)])
def test_luhn(number, valid):
    assert luhn_valid(number) is valid
