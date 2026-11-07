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
# Optional (--distractor-hint): numbers that do not belong to what is asked are a known failure of
# LLMs on word problems (Shi et al. 2023, "Large Language Models Can Be Easily Distracted by Irrelevant
# Context"; arXiv 2406.15444 adds irrelevant numbers and sees ~26% relative drops). Domain-neutral wording.
# Not yet measured here: compare calc-gorilla and calc-gorilla-hint on the same 420 items.
DISTRACTOR_HINT = ("\n수식에는 질문이 묻는 값에 실제로 쓰이는 숫자만 넣으세요. "
                   "'제외', '빼고', '별도'라고 한 항목이나 값과 관계없는 숫자(번호, 연도, 배송비 등)는 넣지 마세요.")
RETRY_PREFILL = "계산식: "
# Optional (--direction-retry): sent only when direction_flag fired on a rate-of-change formula.
DIRECTION_HINT = ("\n증감률·성장률은 (나중 값 - 처음 값) / 처음 값입니다. 나누는 수는 비교의 출발점인 처음(기준) 값이고, "
                  "값이 줄었으면 결과는 음수입니다.")
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

    A number counts as present when it is in the question as written, or as its percent form
    (``5`` in the question, ``0.05`` in the formula, or the reverse). It still cannot tell a
    wrong-but-present number from the right one: that is what ``direction_flag`` is for.
    """
    asked = _numbers(question)
    present = asked | _FREE  # 1 and 100 are structure (growth = x/y - 1, percent = * 100)
    present |= {round(n / 100, 10) for n in asked} | {round(n * 100, 10) for n in asked}
    return sorted(n for n in _numbers(clean(expression)) if round(n, 10) not in present)


_RATE = re.compile(r"증감률|증가율|성장률|변화율|변동률|증가폭")  # not 감소율: (old - new) / old is right there
_DELTA_OVER = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\)\s*/\s*(\d+(?:\.\d+)?)")


def direction_flag(question: str, expression: str) -> bool:
    """True when a signed rate-of-change question gets ``(a - b) / a``: the minuend is also the base.

    ``(new - old) / old`` is right, so ``(old - new) / old`` has the sign reversed (a fall comes out
    positive) and ``(new - old) / new`` has the wrong base. Only this one shape is checked, so a False
    says nothing. Questions that ask for a fall (감소율) are skipped: there ``(old - new) / old`` is right.
    """
    if not _RATE.search(question):
        return False
    flat = clean(expression).replace(",", "")
    return any(float(a) == float(c) and float(a) != float(b) for a, b, c in _DELTA_OVER.findall(flat))


def question_text(item: dict) -> str:
    return "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")


def looked_up(question: str, reply: str) -> bool:
    """True when the ``정답:`` value is made only of numbers that appear in the question (read off a table).

    A number that is not in the question was worked out in the model's head, which is exactly what the
    calculator is there to prevent. An answer without any number is not a calculation either.
    """
    answers = _ANSWER.findall(reply)
    if not answers:
        return False
    values = _numbers(answers[-1])
    return values <= _numbers(question)  # an answer with no digits (a name, a label) needs no calculator


class CalcPipeline:
    """Wraps any generator (local model or endpoint) of ``genesis.harness.evaluate``."""

    name = "gorilla"

    def __init__(self, base, digits: int = 2, retry: bool = True, hint: bool = False,
                 prefill: bool = False, direction_retry: bool = False):
        self.base, self.digits, self.retry, self.hint, self.prefill = base, digits, retry, hint, prefill
        self.direction_retry = direction_retry
        self.direction_rescued = 0
        self.calls = self.fallbacks = self.ungrounded = self.direction = self.retried = self.rescued = 0

    def __getattr__(self, attribute):  # hardware, precision, use_cuda, sync ...
        return getattr(self.base, attribute)

    def _check_grounding(self, item: dict, reply: str) -> bool:
        """Counts suspicious formulas (the answer text stays as the model wrote it); True if the direction is suspect."""
        expression = expression_of(reply)
        if expression is None:
            return False
        question = question_text(item)
        if ungrounded(question, expression):
            self.ungrounded += 1
        flagged = direction_flag(question, expression)
        self.direction += flagged
        return flagged

    def _retry_item(self, item: dict, direction: bool = False) -> dict:
        clone = with_instruction(item, RETRY_INSTRUCTION + (DISTRACTOR_HINT if self.hint else "")
                                 + (DIRECTION_HINT if direction else ""))
        if self.prefill:
            clone["prefill"] = RETRY_PREFILL  # local generators force the reply to start with it
        return clone

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        outputs = self.base.generate([with_instruction(item, INSTRUCTION + (DISTRACTOR_HINT if self.hint else "")) for item in batch])
        results: list[list] = []  # [final text, tokens, used the calculator]
        again: list[int] = []
        flagged: set[int] = set()  # answers with a formula whose direction looks wrong (--direction-retry)
        for index, (item, (text, n_tokens)) in enumerate(zip(batch, outputs)):
            final, used = finish(text, self.digits)
            self.calls += 1
            if used:
                if self._check_grounding(item, text) and self.direction_retry:
                    again.append(index)
                    flagged.add(index)
            elif self.retry and not looked_up(question_text(item), text):
                again.append(index)
            results.append([final, n_tokens, used])
        if again:
            second = self.base.generate([self._retry_item(batch[i], i in flagged) for i in again])
            for index, (text, n_tokens) in zip(again, second):
                self.retried += 1
                results[index][1] += n_tokens  # the retry costs tokens too
                final, used = finish(text, self.digits)
                if not used:
                    continue
                if index in flagged:  # keep the first answer unless the retry fixed the direction
                    if not self._check_grounding(batch[index], text):
                        self.direction_rescued += 1
                        results[index][0] = final
                    continue
                self.rescued += 1
                self._check_grounding(batch[index], text)
                results[index][0], results[index][2] = final, True
        self.fallbacks += sum(1 for _, _, used in results if not used)
        return [(final, n_tokens) for final, n_tokens, _ in results]
