"""Gorilla: let the model write the formula and let a calculator do the arithmetic.

The model is asked to answer with one line ``계산식: <expression>`` (numbers taken from the
question and the four operations only). The pipeline evaluates that line with
``genesis.tools.calculate`` and appends ``정답: <value>``. Two escape hatches are closed:
a formula typed on the ``정답:`` line is still calculated, and an answer that is neither a formula
nor a value copied from the question (a number worked out in the model's head) gets one retry that
asks for the formula only. If the model still gives no usable expression the original reply is kept,
so a failure shows up in the score.

Domain-neutral: nothing here knows what the numbers mean.
"""
from __future__ import annotations

import copy
import re

from genesis.tools import CalculationError, calculate, clean, format_number

INSTRUCTION = ("\n\n수치를 계산해야 하면 직접 계산하지 말고, 마지막 줄에 '계산식: <수식>' 한 줄만 쓰세요. "
               "수식에는 질문의 숫자와 + - * / ( ) 만 사용하고, 단위(%, 원)와 '정답'은 쓰지 마세요. "
               "계산이 필요 없고 표에서 값을 그대로 찾는 질문이면 '계산식'을 쓰지 말고 '정답: 값'을 쓰세요.")
RETRY_INSTRUCTION = ("\n\n직접 계산하지 말고 마지막 줄에 '계산식: <수식>' 한 줄만 쓰세요. "
                     "수식에는 질문의 숫자와 + - * / ( ) 만 사용하고, 단위(%, 원)와 '정답'은 쓰지 마세요.")
_EXPRESSION = re.compile(r"(?:계산식|Expression|式)\s*[:：]\s*(.+)", re.IGNORECASE)
_ANSWER = re.compile(r"정답\s*[:：]\s*(.+)")
# arithmetic between numbers on an answer line: "134000 × (1 - 0.3)"; a bare "2023-2024" is not a formula
_FORMULA = re.compile(r"\d\s*[+*/×÷]\s*[\d(]|\d\s+[-−]\s+[\d(]")


def with_instruction(item: dict, text: str = INSTRUCTION) -> dict:
    """A copy of the eval item whose last user message carries the gorilla instruction."""
    clone = copy.deepcopy(item)
    for message in reversed(clone["messages"]):
        if message["role"] == "user":
            message["content"] += text
            break
    return clone


def expression_of(reply: str) -> str | None:
    """The formula to calculate: the last ``계산식:`` line, else a formula typed on the ``정답:`` line."""
    found = _EXPRESSION.findall(reply)
    if found:
        return found[-1]
    typed = [answer for answer in _ANSWER.findall(reply) if _FORMULA.search(answer)]
    return typed[-1] if typed else None


def finish(reply: str, digits: int = 2) -> tuple[str, bool]:
    """(final text, used_calculator). Keeps the reply unchanged when no expression can be evaluated."""
    expression = expression_of(reply)
    if expression is None:
        return reply, False
    try:
        value = calculate(expression)
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


def looked_up(question: str, reply: str) -> bool:
    """True when the ``정답:`` value is made only of numbers that appear in the question (read off a table).

    A number that is not in the question was worked out in the model's head, which is exactly what the
    calculator is there to prevent.
    """
    answers = _ANSWER.findall(reply)
    if not answers:
        return False
    values = _numbers(answers[-1])
    return bool(values) and values <= _numbers(question)


class CalcPipeline:
    """Wraps any generator (local model or endpoint) of ``genesis.harness.evaluate``."""

    name = "gorilla"

    def __init__(self, base, digits: int = 2, retry: bool = True):
        self.base, self.digits, self.retry = base, digits, retry
        self.calls = self.fallbacks = self.ungrounded = self.retried = self.rescued = 0

    def __getattr__(self, attribute):  # hardware, precision, use_cuda, sync ...
        return getattr(self.base, attribute)

    def _check_grounding(self, item: dict, reply: str) -> None:
        expression = expression_of(reply)
        if expression is not None and ungrounded(question_text(item), expression):
            self.ungrounded += 1  # counted only: the answer text stays as the model produced it

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        outputs = self.base.generate([with_instruction(item) for item in batch])
        results: list[list] = []  # [final text, tokens, used the calculator]
        again: list[int] = []
        for index, (item, (text, n_tokens)) in enumerate(zip(batch, outputs)):
            final, used = finish(text, self.digits)
            self.calls += 1
            if used:
                self._check_grounding(item, text)
            elif self.retry and not looked_up(question_text(item), text):
                again.append(index)
            results.append([final, n_tokens, used])
        if again:
            second = self.base.generate([with_instruction(batch[i], RETRY_INSTRUCTION) for i in again])
            for index, (text, n_tokens) in zip(again, second):
                self.retried += 1
                results[index][1] += n_tokens  # the retry costs tokens too
                final, used = finish(text, self.digits)
                if used:
                    self.rescued += 1
                    self._check_grounding(batch[index], text)
                    results[index][0], results[index][2] = final, True
        self.fallbacks += sum(1 for _, _, used in results if not used)
        return [(final, n_tokens) for final, n_tokens, _ in results]
