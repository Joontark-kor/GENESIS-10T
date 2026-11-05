"""Gorilla: let the model write the formula and let a calculator do the arithmetic.

The model is asked to answer with one line ``계산식: <expression>`` (numbers taken from the
question and the four operations only). The pipeline evaluates that line with
``genesis.tools.calculate`` and appends ``정답: <value>``. If the model gives no usable
expression the original reply is kept, so a failure still shows up in the score.

Domain-neutral: nothing here knows what the numbers mean.
"""
from __future__ import annotations

import copy
import re

from genesis.tools import CalculationError, calculate, clean, format_number

INSTRUCTION = ("\n\n수치를 계산해야 하면 직접 계산하지 말고, 마지막 줄에 '계산식: <수식>' 한 줄만 쓰세요. "
               "수식에는 질문의 숫자와 + - * / ( ) 만 사용하고, 단위(%, 원)와 '정답'은 쓰지 마세요. "
               "계산이 필요 없고 표에서 값을 그대로 찾는 질문이면 '계산식'을 쓰지 말고 '정답: 값'을 쓰세요.")
_EXPRESSION = re.compile(r"(?:계산식|Expression|式)\s*[:：]\s*(.+)", re.IGNORECASE)


def with_instruction(item: dict) -> dict:
    """A copy of the eval item whose last user message carries the gorilla instruction."""
    clone = copy.deepcopy(item)
    for message in reversed(clone["messages"]):
        if message["role"] == "user":
            message["content"] += INSTRUCTION
            break
    return clone


def finish(reply: str, digits: int = 2) -> tuple[str, bool]:
    """(final text, used_calculator). Keeps the reply unchanged when no expression can be evaluated."""
    found = _EXPRESSION.findall(reply)
    if not found:
        return reply, False
    try:
        value = calculate(found[-1])
    except CalculationError:
        return reply, False
    return f"{reply.rstrip()}\n정답: {format_number(value, digits)}", True


_FREE = {1.0, 100.0}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text.replace(",", ""))}


def ungrounded(question: str, expression: str) -> list[float]:
    """Numbers in the formula that do not appear in the question (a made-up or mis-copied figure).

    The check is deliberately dumb: it cannot tell a wrong-but-present number from the right one,
    so it catches invented figures, not picked-the-wrong-row mistakes.
    """
    present = _numbers(question) | _FREE  # 1 and 100 are structure (growth = x/y - 1, percent = * 100)
    return sorted(n for n in _numbers(clean(expression)) if n not in present)


def question_text(item: dict) -> str:
    return "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")


class CalcPipeline:
    """Wraps any generator (local model or endpoint) of ``genesis.harness.evaluate``."""

    name = "gorilla"

    def __init__(self, base, digits: int = 2):
        self.base, self.digits = base, digits
        self.calls = self.fallbacks = self.ungrounded = 0

    def __getattr__(self, attribute):  # hardware, precision, use_cuda, sync ...
        return getattr(self.base, attribute)

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        outputs = self.base.generate([with_instruction(item) for item in batch])
        results = []
        for item, (text, n_tokens) in zip(batch, outputs):
            final, used = finish(text, self.digits)
            if used:
                bad = ungrounded(question_text(item), _EXPRESSION.findall(text)[-1])
                if bad:
                    self.ungrounded += 1  # counted only: the answer text stays as the model produced it
            self.calls += 1
            self.fallbacks += 0 if used else 1
            results.append((final, n_tokens))
        return results
