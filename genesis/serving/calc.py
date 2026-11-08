"""Ask a served model a numeric question and get back an answer you can check.

The model writes the formula, ``genesis.tools.calculate`` does the arithmetic (the "polarbear" structure,
see docs/12_polarbear_ko.md). Unlike the evaluation harness, an app needs more than a score: it needs to
see *which formula* produced the number, and whether anything looked off. ``CalcResult`` carries both.

    python -m genesis.serving.calc --url http://localhost:8000/v1 --model fin "정가 12,000원, 15% 할인가는?"

Only talks to an OpenAI-compatible endpoint (``EndpointClient``); no model code is imported.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

from genesis.harness.polarbear import (CalcPipeline, direction_flag, expression_of, looked_up, percent_flag,
                                     prose_mismatch, question_text, ungrounded)
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

    def __init__(self, client: EndpointClient, model: str, max_tokens: int, template_kwargs: dict | None,
                 prefill: bool = False):
        self.client, self.model, self.max_tokens, self.template_kwargs = client, model, max_tokens, template_kwargs
        self.prefill = prefill

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        out = []
        for item in batch:
            messages, extra, start = item["messages"], {}, ""
            if self.prefill and item.get("prefill"):
                # the retry starts the reply for the model (vLLM / mini_server: continue_final_message), so it
                # cannot answer in its head; the server returns only what follows the prefill
                start = item["prefill"]
                messages = messages + [{"role": "assistant", "content": start}]
                extra = {"continue_final_message": True, "add_generation_prompt": False}
            reply = self.client.chat(self.model, messages, max_tokens=self.max_tokens,
                                     template_kwargs=self.template_kwargs, **extra)
            out.append((start + reply.text, reply.completion_tokens))
        return out


class CalcClient:
    def __init__(self, client: EndpointClient, model: str, digits: int = 2, retry: bool = True,
                 hint: bool = False, max_tokens: int = 160, template_kwargs: dict | None = None,
                 prefill: bool = False, unit_hint: bool = False):
        # unit_hint=True tells the model to multiply by 100 when a percent is asked (CAGR gave 0.21 instead of 21)
        # prefill=True needs a server that continues a final assistant message (vLLM, genesis mini_server);
        # Ollama and others ignore it, so it is off by default
        self.pipeline = CalcPipeline(_Generator(client, model, max_tokens, template_kwargs, prefill),
                                     digits=digits, retry=retry, hint=hint, prefill=prefill, unit_hint=unit_hint)
        self.digits = digits

    def ask(self, messages: list[dict]) -> CalcResult:
        item = {"messages": messages}
        before = self.pipeline.retried
        text, tokens = self.pipeline.generate([item])[0]
        formula = expression_of(text)
        answer, warnings = None, []
        if formula is not None:
            try:
                value = calculate(formula)
                answer = format_number(value, self.digits)
            except CalculationError:
                warnings.append("식을 계산하지 못했습니다")
            else:
                missing = ungrounded(question_text(item), formula)
                if missing:
                    warnings.append("질문에 없는 숫자가 식에 있습니다: " + ", ".join(f"{n:g}" for n in missing))
                if direction_flag(question_text(item), formula):
                    warnings.append("증감률 식의 방향(부호)을 확인하세요")
                said = prose_mismatch(question_text(item), text, value)
                if said is not None:
                    warnings.append(f"설명에서는 {said:g}이라고 했는데 식의 계산 결과는 {answer}입니다: 식에 단계가 빠졌을 수 있습니다")
                if percent_flag(question_text(item), formula):
                    warnings.append("%를 물었는데 식에 100(또는 %)이 없습니다: 소수로 나왔을 수 있습니다")
        elif not looked_up(question_text(item), text):  # an answer read off the question, or a name, needs no formula
            warnings.append("식 없이 답했습니다(계산기를 쓰지 않음)")
        return CalcResult(text=text, answer=answer, formula=formula, used_calculator=answer is not None,
                          retried=self.pipeline.retried > before, warnings=warnings, completion_tokens=tokens)


def main(argv: list[str] | None = None) -> CalcResult:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--distractor-hint", action="store_true")
    p.add_argument("--no-retry", action="store_true")
    p.add_argument("--unit-hint", action="store_true", help="ask for percent values as 25, not 0.25")
    p.add_argument("--prefill", action="store_true", help="retry with '계산식:' already written (vLLM, mini_server)")
    p.add_argument("question")
    args = p.parse_args(argv)
    result = CalcClient(EndpointClient(args.url), args.model, retry=not args.no_retry,
                        hint=args.distractor_hint, prefill=args.prefill, unit_hint=args.unit_hint).ask([{"role": "user", "content": args.question}])
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
