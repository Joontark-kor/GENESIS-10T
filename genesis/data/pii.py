"""Regex masking for common Korean personal identifiers.

This is a *first line of defence*, not a guarantee: it does not catch names,
addresses, bank account numbers (formats vary per bank and collide with other
numbers) or identifiers written in words. Sample and review masked output, and
keep sources with heavy personal data out of the corpus altogether.
"""
from __future__ import annotations

import re

_RRN = re.compile(  # 주민등록번호 / 외국인등록번호: YYMMDD-GNNNNNN
    r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\s?-\s?[1-8]\d{6}(?!\d)")
_MOBILE = re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")
_LANDLINE = re.compile(  # 02-123-4567, 031)123-4567, 070-1234-5678 (separators required)
    r"(?<![\d-])0(?:2|[3-6][1-5]|70)[-.)]\s?\d{3,4}[-.]\d{4}(?!\d)")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<!\d)(?:\d{4}[- ]){3}\d{4}(?!\d)")

PLACEHOLDERS = {
    "rrn": "[주민번호]",
    "phone": "[전화번호]",
    "email": "[이메일]",
    "card": "[카드번호]",
}


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace detected identifiers with placeholders; return counts per kind."""
    counts = {kind: 0 for kind in PLACEHOLDERS}

    def sub(pattern: re.Pattern, kind: str, check=None):
        nonlocal text

        def repl(match: re.Match) -> str:
            if check is not None and not check(match.group(0)):
                return match.group(0)
            counts[kind] += 1
            return PLACEHOLDERS[kind]

        text = pattern.sub(repl, text)

    sub(_RRN, "rrn")              # before phone: an RRN's digits could look like a number
    sub(_CARD, "card", _luhn_ok)
    sub(_EMAIL, "email")
    sub(_MOBILE, "phone")
    sub(_LANDLINE, "phone")
    return text, counts
