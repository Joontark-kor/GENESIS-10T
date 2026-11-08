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
# Optional (--exclude-check): the model first names the question's numbers it will NOT use ("제외: 3, 2024"),
# then writes the formula. A formula that still contains a named number contradicts itself (seen on the hard
# set: "수수료는 별도라 포함하지 않는다" and then "... - 4" in the formula), so it gets one retry that names them.
EXCLUDE_HINT = ("\n계산식 바로 앞 줄에 '제외: 숫자, 숫자' 형식으로, 질문에 나오지만 식에 넣지 않을 숫자를 먼저 적으세요"
                "(없으면 '제외: 없음').")
_EXCLUDED = re.compile(r"제외\s*[:：]\s*(.+)")
# Optional (--unit-hint): on the hard set CAGR was 0% because the formula gave a fraction (0.37) when the
# question asked for a percent (37.12). Domain-neutral: the answer must be in the unit that was asked for.
UNIT_HINT = ("\n답의 단위가 %이면 식의 결과가 퍼센트 값 그대로(0.25가 아니라 25)가 되도록 100을 곱하세요. "
             "질문이 요구한 단위로 답이 나와야 합니다.")
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


_REF = re.compile(r"#(\d+)")
# Optional (CalcClient multistep=True): several ``계산식:`` lines, a later one may use ``#1`` for the first line's value.
MULTISTEP_HINT = ("\n계산이 여러 단계이면 '계산식: <수식>' 줄을 단계마다 한 줄씩 쓰고, 앞 단계의 결과는 #1, #2처럼 "
                  "번호로 가져다 쓰세요. 마지막 계산식 줄이 최종 답입니다.")


def _resolved(lines: list[str]) -> str:
    """The last formula with ``#n`` replaced by the n-th formula (in parentheses), so one calculation covers all steps."""
    done: list[str] = []
    for line in lines:
        done.append(_REF.sub(lambda m: f"({done[int(m.group(1)) - 1]})" if 0 < int(m.group(1)) <= len(done) else m.group(0), line))
    return done[-1]


def expression_of(reply: str) -> str | None:
    """The formula to calculate: the last ``계산식:`` line (steps joined), else a formula typed on the ``정답:`` line."""
    found = _EXPRESSION.findall(reply)
    if found:
        return _resolved(found)
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
_SCALE = {"천": 1e3, "만": 1e4, "억": 1e8, "조": 1e12}
_SCALED = re.compile(r"(\d+(?:\.\d+)?)\s*(천|만|억|조)")
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
    for number, unit in _SCALED.findall(question.replace(",", "")):  # "1,000만원" is 10000000 in a formula
        present.add(round(float(number) * _SCALE[unit], 10))
    flat = re.sub(r"\*\*\s*\(?\s*[23]\s*\)?(?![\d.])", "", clean(expression))  # a square or cube is structure, not a made-up figure
    return sorted(n for n in _numbers(flat) if round(n, 10) not in present)


_ASKS_PERCENT = re.compile(r"몇\s*(?:%|퍼센트|프로)")


def percent_flag(question: str, expression: str) -> bool:
    """True when the question asks "몇 %" but the formula has no ``100`` and no ``%``: it likely returns a fraction.

    Only this shape is checked, so a False says nothing (the formula may reach 100 some other way).
    """
    if not _ASKS_PERCENT.search(question):
        return False
    if re.search(r"-\s*1\s*\*\s*100\s*$", expression):  # "x/y - 1*100" multiplies only the 1: missing parentheses
        return True
    return "100" not in expression and "%" not in expression


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


def contradiction(reply: str, expression: str) -> list[float]:
    """Numbers the reply said it would leave out (``제외:`` line) that the formula uses anyway."""
    lines = _EXCLUDED.findall(reply)
    if not lines:
        return []
    skipped = _numbers(lines[-1]) - _FREE
    return sorted(skipped & _numbers(clean(expression)))


def prose_mismatch(question: str, reply: str, value: float) -> float | None:
    """The number the reply's own prose arrived at when it differs from the calculated ``value``, else None.

    Seen on the multi-step set: the prose solved the problem (34.71) but the ``계산식:`` line held only the
    first step (3749). The last number in the text before the formula that is not in the question is the
    prose's answer; if the calculator disagrees, the formula probably dropped a step. A signal only
    (the prose can also stop at an intermediate figure), so it is a warning, not a retry.
    """
    marker = _EXPRESSION.search(reply)
    prose = reply[:marker.start()] if marker else reply
    asked = _numbers(question)
    figures = [n for n in (float(m) for m in _NUMBER.findall(prose.replace(",", ""))) if n not in asked]
    if not figures:
        return None
    said = figures[-1]
    return said if abs(said - value) > max(0.01, abs(value) * 1e-3) else None


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
                 prefill: bool = False, direction_retry: bool = False, unit_hint: bool = False,
                 exclude_check: bool = False, multistep: bool = False):
        self.unit_hint, self.exclude_check, self.multistep = unit_hint, exclude_check, multistep
        self.contradicted = self.contradiction_rescued = 0
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

    def _extra(self) -> str:
        return ((DISTRACTOR_HINT if self.hint else "") + (UNIT_HINT if self.unit_hint else "")
                + (MULTISTEP_HINT if self.multistep else "")
                + (EXCLUDE_HINT if self.exclude_check else ""))

    def _retry_item(self, item: dict, note: str = "") -> dict:
        clone = with_instruction(item, RETRY_INSTRUCTION + self._extra() + note)
        if self.prefill:
            clone["prefill"] = RETRY_PREFILL  # local generators force the reply to start with it
        return clone

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        outputs = self.base.generate([with_instruction(item, INSTRUCTION + self._extra()) for item in batch])
        results: list[list] = []  # [final text, tokens, used the calculator]
        again: list[int] = []
        flagged: dict[int, str] = {}  # formulas to redo with a note: wrong direction / contradicts its own 제외 line
        for index, (item, (text, n_tokens)) in enumerate(zip(batch, outputs)):
            final, used = finish(text, self.digits)
            self.calls += 1
            if used:
                wrong_way = self._check_grounding(item, text)
                clash = contradiction(text, expression_of(text) or "") if self.exclude_check else []
                self.contradicted += bool(clash)
                if clash:
                    note = f"\n식에 {', '.join(f'{n:g}' for n in clash)}을(를) 넣지 마세요. 제외한 숫자입니다."
                    again.append(index)
                    flagged[index] = note
                elif wrong_way and self.direction_retry:
                    again.append(index)
                    flagged[index] = DIRECTION_HINT
            elif self.retry and not looked_up(question_text(item), text):
                again.append(index)
            results.append([final, n_tokens, used])
        if again:
            second = self.base.generate([self._retry_item(batch[i], flagged.get(i, '')) for i in again])
            for index, (text, n_tokens) in zip(again, second):
                self.retried += 1
                results[index][1] += n_tokens  # the retry costs tokens too
                final, used = finish(text, self.digits)
                if not used:
                    continue
                if index in flagged:  # keep the first answer unless the retry fixed what was flagged
                    again_wrong = self._check_grounding(batch[index], text)
                    clash = contradiction(text, expression_of(text) or "") if self.exclude_check else []
                    if not clash and not (again_wrong and flagged[index] == DIRECTION_HINT):
                        if flagged[index] == DIRECTION_HINT:
                            self.direction_rescued += 1
                        else:
                            self.contradiction_rescued += 1
                        results[index][0] = final
                    continue
                self.rescued += 1
                self._check_grounding(batch[index], text)
                results[index][0], results[index][2] = final, True
        self.fallbacks += sum(1 for _, _, used in results if not used)
        return [(final, n_tokens) for final, n_tokens, _ in results]
