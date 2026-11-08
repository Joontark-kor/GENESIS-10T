from genesis.serving.calc import CalcClient
from genesis.serving.endpoint import Reply


class FakeEndpoint:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def chat(self, model, messages, max_tokens=256, temperature=0.0, template_kwargs=None, **extra):
        self.calls.append(messages[-1]["content"])
        return Reply(text=self.texts.pop(0), completion_tokens=7, prompt_tokens=1, finish_reason="stop")


def ask(*texts, question="정가 12,000원, 15% 할인가는?", **kwargs):
    fake = FakeEndpoint(*texts)
    result = CalcClient(fake, "m", **kwargs).ask([{"role": "user", "content": question}])
    return result, fake


def test_returns_the_formula_and_the_calculated_answer():
    result, _ = ask("계산식: 12000 * (1 - 15%)")
    assert result.formula == "12000 * (1 - 15%)" and result.answer == "10200"
    assert result.used_calculator and not result.retried and result.warnings == []
    assert result.text.endswith("정답: 10200")


def test_retries_once_when_the_model_answers_in_its_head_and_says_so():
    result, fake = ask("정답: 10180", "계산식: 12000 * 0.85")
    assert result.retried and result.answer == "10200" and len(fake.calls) == 2


def test_no_formula_is_reported_not_hidden():
    result, _ = ask("정답: 10180", "정답: 10180")
    assert not result.used_calculator and result.answer is None
    assert any("식 없이" in w for w in result.warnings)


def test_warns_about_numbers_the_question_never_gave():
    result, _ = ask("계산식: 12000 * 0.77")
    assert result.answer == "9240" and any("0.77" in w for w in result.warnings)


def test_warns_about_a_reversed_growth_formula():
    result, _ = ask("계산식: (7240 - 3320) / 7240 * 100", question="3,320에서 7,240으로. 증감률은?")
    assert any("방향" in w for w in result.warnings)
