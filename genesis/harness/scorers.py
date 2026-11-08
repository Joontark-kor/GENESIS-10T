"""Scorers: turn one model reply into ``(prediction, correct)`` for an eval item.

Every item is a JSON object with ``messages`` and ``gold``; the scorer named by
the item's ``scorer`` field (default ``numeric``) decides whether the reply is
right. Domain packages register their own scorers by passing
``--scorer module:function`` to the harness; the core only ships
domain-neutral ones.
"""
from __future__ import annotations

import importlib
import re
from typing import Callable

Scorer = Callable[[str, dict], "tuple[object, bool]"]

_TAGGED = re.compile(r"(?:정답|答え|Answer)\s*[:：]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"[-−△]?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """Number after an answer tag (정답/答え/Answer), else the last number in the text."""
    tagged = _TAGGED.findall(text)
    source = tagged[-1] if tagged else text
    numbers = _NUMBER.findall(source)
    if not numbers:
        return None
    raw = numbers[0] if tagged else numbers[-1]
    negative = raw[0] in "-−△"
    try:
        value = float(raw.lstrip("-−△").replace(",", ""))
    except ValueError:
        return None
    return -value if negative else value


def is_correct(prediction: float | None, gold: float, tolerance: float) -> bool:
    return prediction is not None and abs(prediction - gold) <= tolerance + 1e-9


def numeric(reply: str, item: dict) -> tuple[float | None, bool]:
    """Correct when the parsed number is within the item's ``tolerance`` of ``gold``."""
    prediction = parse_number(reply)
    return prediction, is_correct(prediction, float(item["gold"]), float(item.get("tolerance", 0)))


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def exact(reply: str, item: dict) -> tuple[str, bool]:
    """Correct when the answer (after an answer tag, if any) equals ``gold`` ignoring case and spacing."""
    tagged = _TAGGED.findall(reply)
    answer = (tagged[-1] if tagged else reply).strip()
    return answer, _normalize(answer) == _normalize(item["gold"])


_CHOICE = re.compile(r"[ABCD]")


def choice(reply: str, item: dict) -> tuple[str, bool]:
    """Multiple choice: the first A-D letter after the answer tag (else the first in the reply) must equal ``gold``."""
    tagged = _TAGGED.findall(reply)
    source = (tagged[-1] if tagged else reply).strip().upper()
    letters = _CHOICE.findall(source)
    answer = letters[0] if letters else ""
    return answer, answer == str(item["gold"]).strip().upper()


BUILTIN: dict[str, Scorer] = {"numeric": numeric, "exact": exact, "choice": choice}


def resolve(name: str) -> Scorer:
    """A built-in scorer name, or ``package.module:function`` for a scorer defined elsewhere."""
    if name in BUILTIN:
        return BUILTIN[name]
    if ":" not in name:
        raise ValueError(f"unknown scorer {name!r}; use one of {sorted(BUILTIN)} or module:function")
    module, attr = name.split(":", 1)
    return getattr(importlib.import_module(module), attr)
