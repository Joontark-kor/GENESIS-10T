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

from genesis.tools import CalculationError, calculate, format_number

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


class CalcPipeline:
    """Wraps any generator (local model or endpoint) of ``genesis.harness.evaluate``."""

    name = "gorilla"

    def __init__(self, base, digits: int = 2):
        self.base, self.digits = base, digits
        self.calls = self.fallbacks = 0

    def __getattr__(self, attribute):  # hardware, precision, use_cuda, sync ...
        return getattr(self.base, attribute)

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        outputs = self.base.generate([with_instruction(item) for item in batch])
        results = []
        for text, n_tokens in outputs:
            final, used = finish(text, self.digits)
            self.calls += 1
            self.fallbacks += 0 if used else 1
            results.append((final, n_tokens))
        return results
