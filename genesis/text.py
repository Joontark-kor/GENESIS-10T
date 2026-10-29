"""Text normalization shared by every stage (cleaning, tokenizer training, eval).

Korean text arrives in two Unicode forms: precomposed syllables (NFC, e.g. "한")
and decomposed jamo (NFD, e.g. "ᄒ"+"ᅡ"+"ᆫ"), which macOS file systems and some
scrapers produce. The two look identical but tokenize completely differently,
so every stage normalizes to NFC before anything else.
"""
from __future__ import annotations

import re
import unicodedata

# C0/C1 control characters except tab and newline.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Lossless-in-meaning normalization: NFC, newline unification, control removal."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"


def is_hangul(ch: str) -> bool:
    """Hangul syllables plus compatibility/conjoining jamo."""
    return (is_hangul_syllable(ch) or "ᄀ" <= ch <= "ᇿ"
            or "㄰" <= ch <= "㆏" or "ꥠ" <= ch <= "꥿"
            or "ힰ" <= ch <= "퟿")


def script_counts(text: str) -> dict[str, int]:
    """Counts letters by coarse script, used by language/quality filters."""
    counts = {"hangul": 0, "latin": 0, "han": 0, "kana": 0, "other_letter": 0, "digit": 0}
    for ch in text:
        if is_hangul(ch):
            counts["hangul"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif ch.isalpha():
            if ch.isascii():
                counts["latin"] += 1
            elif "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
                counts["han"] += 1
            elif "぀" <= ch <= "ヿ":
                counts["kana"] += 1
            else:
                counts["other_letter"] += 1
    return counts


def hangul_ratio(text: str) -> float:
    """Share of Hangul among letters (digits and punctuation excluded)."""
    counts = script_counts(text)
    letters = sum(v for k, v in counts.items() if k != "digit")
    return counts["hangul"] / letters if letters else 0.0
