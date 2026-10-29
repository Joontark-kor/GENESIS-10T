"""Heuristic quality filters tuned for Korean web/document text.

Each rule returns a short reason string so the data card can report *why*
documents were dropped. Thresholds are starting points (Gopher/FineWeb-style
heuristics adapted to Hangul); inspect removed samples before trusting them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace
import re

from ..text import hangul_ratio

_WORD = re.compile(r"\w+", re.UNICODE)
_ALLOWED_PUNCT = set(".,!?;:'\"()[]{}<>-–—…·~/%&+=*#@$₩^_|\\`「」『』《》〈〉“”‘’、。・")


@dataclass(frozen=True)
class QualityConfig:
    min_chars: int = 100
    max_chars: int = 500_000
    min_hangul_ratio: float = 0.4          # applied only to language == "ko"
    max_duplicate_line_fraction: float = 0.3
    max_top_bigram_char_fraction: float = 0.2
    max_symbol_fraction: float = 0.25
    max_short_line_fraction: float = 0.67  # lines < 10 chars (menus, nav bars)
    min_lines_for_line_rules: int = 5

    @classmethod
    def permissive(cls) -> "QualityConfig":
        """For tiny hand-made corpora: keep almost everything."""
        return cls(min_chars=1, min_hangul_ratio=0.0, max_duplicate_line_fraction=1.0,
                   max_top_bigram_char_fraction=1.0, max_symbol_fraction=1.0,
                   max_short_line_fraction=1.0)

    def with_overrides(self, overrides: dict) -> "QualityConfig":
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown filter settings: {sorted(unknown)}")
        return replace(self, **overrides)


def duplicate_line_fraction(lines: list[str]) -> float:
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicated = sum(c for c in counts.values() if c > 1)
    return duplicated / len(lines)


def top_bigram_char_fraction(text: str) -> float:
    """Characters covered by the most frequent word bigram (spam/boilerplate signal)."""
    words = _WORD.findall(text)
    if len(words) < 20:  # too short for a stable repetition estimate
        return 0.0
    bigrams = Counter(zip(words, words[1:]))
    (w1, w2), count = bigrams.most_common(1)[0]
    if count < 2:
        return 0.0
    total = sum(len(w) for w in words)
    return count * (len(w1) + len(w2)) / total if total else 0.0


def symbol_fraction(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 1.0
    symbols = sum(1 for ch in visible if not ch.isalnum() and ch not in _ALLOWED_PUNCT)
    return symbols / len(visible)


def quality_reason(text: str, language: str, config: QualityConfig) -> str | None:
    """Return why ``text`` should be dropped, or ``None`` to keep it."""
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    if language == "ko" and hangul_ratio(text) < config.min_hangul_ratio:
        return "low_hangul_ratio"
    if symbol_fraction(text) > config.max_symbol_fraction:
        return "symbol_heavy"
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= config.min_lines_for_line_rules:
        if duplicate_line_fraction(lines) > config.max_duplicate_line_fraction:
            return "duplicate_lines"
        short = sum(1 for line in lines if len(line) < 10) / len(lines)
        if short > config.max_short_line_fraction:
            return "short_lines"
    if top_bigram_char_fraction(text) > config.max_top_bigram_char_fraction:
        return "repetitive_ngrams"
    return None
