"""Ask a served model a numeric question and get back an answer you can check.

The model writes the formula, ``genesis.tools.calculate`` does the arithmetic (the "gorilla" structure,
see docs/12_gorilla_ko.md). Unlike the evaluation harness, an app needs more than a score: it needs to
see *which formula* produced the number, and whether anything looked off. ``CalcResult`` carries both.

    python -m genesis.serving.calc --url http://localhost:8000/v1 --model fin "정가 12,000원, 15% 할인가는?"

Only talks to an OpenAI-compatible endpoint (``EndpointClient``); no model code is imported.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from genesis.harness.gorilla import (CalcPipeline, direction_flag, expression_of, question_text, ungrounded)
from genesis.serving.endpoint import EndpointClient
from genesis.tools import CalculationError, calculate, format_number


@dataclass
class CalcResult:
    text: str                       # the model's reply, with the calculated ``정답:`` line appended when used
    answer: str | None              # the calculated value as text, None when the calculator was not used
    formula: str | None             # the formula the model wrote (what the app can show or audit)
    used_calculator: bool
    retried: bool                   # the first reply had no usable formula and a second try was made
    warnings: list[str] = field(default_factory=list)  # things a careful app should look at before trusting it
    completion_tokens: int = 0


class _Generator:
    """The harness pipeline talks to a ``generate(batch)`` object; this one calls the endpoint."""

    def __init__(self, client: EndpointClient, model: str, max_tokens: int, template_kwargs: dict | None):
        self.client, self.model, self.max_tokens, self.template_kwargs = client, model, max_tokens, template_kwargs

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        out = []
        for item in batch:
            reply = self.client.chat(self.model, item["messages"], max_tokens=self.max_tokens,
                                     template_kwargs=self.template_kwargs)
            out.append((reply.text, reply.completion_tokens))
        return out


class CalcClient:
    def __init__(self, client: EndpointClient, model: str, digits: int = 2, retry: bool = True,
                 hint: bool = False, max_tokens: int = 160, template_kwargs: dict | None = None):
        # prefill needs local generation, so it is not offered here (an endpoint ignores it)
        self.pipeline = CalcPipeline(_Generator(client, model, max_tokens, template_kwargs),
                                     digits=digits, retry=retry, hint=hint)
        self.digits = digits

    def ask(self, messages: list[dict]) -> CalcResult:
        item = {"messages": messages}
        before = self.pipeline.retried
        text, tokens = self.pipeline.generate([item])[0]
        formula = expression_of(text)
        answer, warnings = None, []
        if formula is not None:
            try:
                answer = format_number(calculate(formula), self.digits)
            except CalculationError:
                warnings.append("식을 계산하지 못했습니다")
            else:
                missing = ungrounded(question_text(item), formula)
                if missing:
                    warnings.append("질문에 없는 숫자가 식에 있습니다: " + ", ".join(f"{n:g}" for n in missing))
                if direction_flag(question_text(item), formula):
                    warnings.append("증감률 식의 방향(부호)을 확인하세요")
        else:
            warnings.append("식 없이 답했습니다(계산기를 쓰지 않음)")
        return CalcResult(text=text, answer=answer, formula=formula, used_calculator=answer is not None,
                          retried=self.pipeline.retried > before, warnings=warnings, completion_tokens=tokens)


def main(argv: list[str] | None = None) -> CalcResult:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--distractor-hint", action="store_true")
    p.add_argument("--no-retry", action="store_true")
    p.add_argument("question")
    args = p.parse_args(argv)
    result = CalcClient(EndpointClient(args.url), args.model, retry=not args.no_retry,
                        hint=args.distractor_hint).ask([{"role": "user", "content": args.question}])
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
