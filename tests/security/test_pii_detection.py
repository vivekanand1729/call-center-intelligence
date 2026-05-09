"""Security tests: PII detection and redaction."""
from __future__ import annotations

import pytest

from src.security.pii_redactor import redact_pii


# ── Phone formats ──────────────────────────────────────────────────────────────
PHONE_CASES = [
    "Call me at 555-123-4567",
    "My number is (555) 123-4567",
    "Reach me at 555.123.4567",
    "Mobile: 1-555-123-4567",
    "+1 555 123 4567",
]

# ── Email formats ──────────────────────────────────────────────────────────────
EMAIL_CASES = [
    "Email me at john.doe@company.com",
    "Contact support@help-desk.org",
    "user.name+tag@sub.domain.co",
    "simple@example.com",
]

# ── SSN formats ────────────────────────────────────────────────────────────────
SSN_CASES = [
    "My SSN is 123-45-6789",
    "Social security number: 987-65-4321",
]

# ── Credit card formats ────────────────────────────────────────────────────────
CC_CASES = [
    "Card number 1234 5678 9012 3456",
    "My card is 1234-5678-9012-3456",
    "Payment card 1234567890123456",
]

# ── PII embedded in conversation ───────────────────────────────────────────────
EMBEDDED_CASES = [
    ("The customer at john@example.com called 555-123-4567", ["EMAIL", "PHONE"]),
    ("SSN 123-45-6789 and card 4111111111111111 were verified", ["SSN", "CREDIT_CARD"]),
    ("Please send confirmation to user@test.org about order 99", ["EMAIL"]),
]


@pytest.mark.parametrize("text", PHONE_CASES)
def test_phone_redacted(text):
    result = redact_pii(text)
    assert result.pii_found is True
    assert "[REDACTED_PHONE]" in result.redacted_text


@pytest.mark.parametrize("text", EMAIL_CASES)
def test_email_redacted(text):
    result = redact_pii(text)
    assert result.pii_found is True
    assert "[REDACTED_EMAIL]" in result.redacted_text


@pytest.mark.parametrize("text", SSN_CASES)
def test_ssn_redacted(text):
    result = redact_pii(text)
    assert result.pii_found is True
    assert "[REDACTED_SSN]" in result.redacted_text


@pytest.mark.parametrize("text", CC_CASES)
def test_credit_card_redacted(text):
    result = redact_pii(text)
    assert result.pii_found is True
    assert "[REDACTED_CREDIT_CARD]" in result.redacted_text


@pytest.mark.parametrize("text,expected_types", EMBEDDED_CASES)
def test_pii_embedded_in_conversation(text, expected_types):
    result = redact_pii(text)
    assert result.pii_found is True
    for t in expected_types:
        assert f"[REDACTED_{t}]" in result.redacted_text


def test_clean_text_unchanged():
    text = "Hello, how can I help you today?"
    result = redact_pii(text)
    assert result.pii_found is False
    assert result.redacted_text == text


def test_multiple_pii_in_one_text():
    text = "My SSN is 123-45-6789 and email is test@test.com"
    result = redact_pii(text)
    assert "[REDACTED_SSN]" in result.redacted_text
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "123-45-6789" not in result.redacted_text
    assert "test@test.com" not in result.redacted_text


def test_redaction_is_right_to_left_correct():
    """Both SSN and phone in same text — positions must not be corrupted."""
    text = "SSN 123-45-6789 or call 555-867-5309 for help"
    result = redact_pii(text)
    assert result.pii_found is True
    # Original values should not appear
    assert "123-45-6789" not in result.redacted_text
    assert "555-867-5309" not in result.redacted_text


def test_types_found_list():
    text = "SSN 123-45-6789"
    result = redact_pii(text)
    assert "SSN" in result.types_found


def test_empty_text():
    result = redact_pii("")
    assert result.pii_found is False
    assert result.redacted_text == ""


def test_no_raw_pii_after_redaction():
    ssn = "123-45-6789"
    text = f"My social is {ssn}"
    result = redact_pii(text)
    assert ssn not in result.redacted_text
