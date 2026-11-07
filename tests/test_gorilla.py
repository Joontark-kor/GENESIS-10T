import pytest

from genesis.harness.gorilla import CalcPipeline, finish, with_instruction
from genesis.harness.scorers import numeric
from genesis.tools import CalculationError, calculate, format_number


def test_calculate_is_exact_and_plain():
    assert calculate("(2,248,743 - 2,363,057) / 2,363,057 * 100") == pytest.approx(-4.8372, abs=1e-3)
    assert calculate("1,361,965 ÷ 2,197,825 × 100 = 62.00%") == pytest.approx(61.97, abs=1e-2)
    assert calculate("-3 + +2 ** 2") == 1
    assert format_number(-4.8372) == "-4.84" and format_number(11.0) == "11" and format_number(61.97) == "61.97"


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "abs(3)", "a + 1", "1 / 0", "9 ** 99",
                                 "[1, 2]", "", "1 +", "(1).real", "x" * 500])
def test_calculate_refuses_everything_else(bad):
    with pytest.raises(CalculationError):
        calculate(bad)


def test_finish_uses_the_calculator_or_keeps_the_reply():
    final, used = finish("식을 세우면\n계산식: (305,515 - 186,852) / 186,852 * 100")
    assert used and final.endswith("정답: 63.51")
    assert numeric(final, {"gold": 63.51, "tolerance": 0.01})[1]
    same, used = finish("정답: 12.0")
    assert not used and same == "정답: 12.0"
    broken, used = finish("계산식: 나누기(3, 4)")
    assert not used and broken == "계산식: 나누기(3, 4)"


def test_pipeline_wraps_a_generator_and_counts_fallbacks():
    seen = []

    class Fake:
        hardware = "fake"

        def sync(self):
            pass

        def generate(self, batch):
            seen.extend(item["messages"][-1]["content"] for item in batch)
            return [("계산식: 10 / 4", 7), ("모르겠습니다", 3)]

    item = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], "gold": 2.5}
    pipeline = CalcPipeline(Fake(), retry=False)
    out = pipeline.generate([item, item])
    assert out[0] == ("계산식: 10 / 4\n정답: 2.5", 7) and out[1] == ("모르겠습니다", 3)
    assert pipeline.hardware == "fake" and pipeline.fallbacks == 1 and pipeline.calls == 2
    assert all(text.startswith("q") and "계산식" in text for text in seen)
    assert item["messages"][-1]["content"] == "q"  # the original item is not modified
    assert "계산식" in with_instruction(item)["messages"][-1]["content"]


def test_ungrounded_finds_numbers_missing_from_the_question():
    from genesis.harness.gorilla import ungrounded
    question = "당기순이익 10,682백만원, 자기자본 106,826백만원일 때 ROE는?"
    assert ungrounded(question, "10682 / 106826 * 100") == []
    assert ungrounded(question, "10682.6 / 106826 * 100") == [10682.6]


def test_pipeline_counts_ungrounded_without_changing_the_answer():
    from genesis.harness.gorilla import CalcPipeline

    class Fake:
        def generate(self, batch):
            return [("계산식: 5 / 2", 3)]

    item = {"messages": [{"role": "user", "content": "A는 5, B는 3이다."}]}
    pipeline = CalcPipeline(Fake())
    out = pipeline.generate([item])
    assert out[0][0].endswith("정답: 2.5") and pipeline.ungrounded == 1


def test_percent_sign_inside_a_formula_means_divide_by_100():
    assert calculate("4800000 * 3% * 4") == pytest.approx(576000)
    assert calculate("(1361965 / 2197825) * 100%") == pytest.approx(61.97, abs=1e-2)  # trailing 100% stays x100
    assert calculate("200 * (1 + 5%)") == pytest.approx(210)


def test_formula_typed_on_the_answer_line_is_still_calculated():
    final, used = finish("정답: 134000 × (1 - 0.3)")  # seen in the 2026-11-06 Colab run
    assert used and final.endswith("정답: 93800")
    final, used = finish("정답: 2023-2024")  # a range is not a formula
    assert not used and final == "정답: 2023-2024"
    final, used = finish("정답: 163,200")
    assert not used


def test_looked_up_tells_a_table_value_from_a_number_worked_out_in_the_head():
    from genesis.harness.gorilla import looked_up
    question = "매출액 106,826백만원, 영업이익 10,682백만원이다."
    assert looked_up(question, "정답: 106,826")
    assert not looked_up(question, "정답: 10")  # not in the question: computed
    assert not looked_up(question, "모르겠습니다")  # no answer line
    assert looked_up(question, "정답: 삼성전자")  # a name is not a calculation: never retried


class _Scripted:
    """Answers the first call with `first`, every later call with `second`."""

    def __init__(self, first, second):
        self.first, self.second, self.calls = first, second, []

    def generate(self, batch):
        self.calls.append([item["messages"][-1]["content"] for item in batch])
        replies = self.first if len(self.calls) == 1 else self.second
        return replies[:len(batch)]


def test_retry_rescues_an_answer_worked_out_without_a_formula():
    item = {"messages": [{"role": "user", "content": "134,000원의 30% 할인가는?"}]}
    base = _Scripted([("정답: 93900", 5)], [("계산식: 134000 * (1 - 30%)", 9)])
    pipeline = CalcPipeline(base)
    out = pipeline.generate([item])
    assert out[0][0].endswith("정답: 93800") and out[0][1] == 14  # tokens of both calls
    assert pipeline.retried == 1 and pipeline.rescued == 1 and pipeline.fallbacks == 0
    assert "직접 계산하지 말고" in base.calls[1][0]


def test_retry_skips_table_lookups_and_keeps_the_reply_when_it_fails():
    lookup = {"messages": [{"role": "user", "content": "매출액은 106,826백만원이다. 매출액은?"}]}
    worked = {"messages": [{"role": "user", "content": "A는 8, B는 4이다. A÷B는?"}]}
    base = _Scripted([("정답: 106,826", 4), ("정답: 2", 4)], [("모르겠습니다", 2)])
    pipeline = CalcPipeline(base)
    out = pipeline.generate([lookup, worked])
    assert out[0][0] == "정답: 106,826"  # lookup: no retry
    assert out[1] == ("정답: 2", 6)  # retried, no formula came back: original reply kept, tokens still counted
    assert len(base.calls) == 2 and len(base.calls[1]) == 1
    assert pipeline.retried == 1 and pipeline.rescued == 0
    assert pipeline.fallbacks == 2  # "no calculator used": the lookup and the failed retry


def test_retry_can_be_switched_off():
    item = {"messages": [{"role": "user", "content": "A는 8, B는 4이다. A÷B는?"}]}
    base = _Scripted([("정답: 2", 4)], [("계산식: 8 / 4", 4)])
    pipeline = CalcPipeline(base, retry=False)
    assert pipeline.generate([item])[0] == ("정답: 2", 4) and len(base.calls) == 1


def test_known_limit_an_answer_equal_to_a_number_in_the_question_is_not_retried():
    from genesis.harness.gorilla import looked_up
    assert looked_up("A는 6, B는 3이다. A÷B는?", "정답: 3")  # 6/3 = 2 would be caught; this guess is not


def test_distractor_hint_is_opt_in_and_reaches_first_pass_and_retry():
    from genesis.harness.gorilla import DISTRACTOR_HINT
    item = {"messages": [{"role": "user", "content": "A는 8, B는 4이다. A÷B는?"}]}
    plain = _Scripted([("정답: 2", 1)], [("계산식: 8 / 4", 1)])
    CalcPipeline(plain).generate([item])
    assert all(DISTRACTOR_HINT not in text for call in plain.calls for text in call)
    hinted = _Scripted([("정답: 2", 1)], [("계산식: 8 / 4", 1)])
    CalcPipeline(hinted, hint=True).generate([item])
    assert len(hinted.calls) == 2 and all(DISTRACTOR_HINT in call[0] for call in hinted.calls)


def test_retry_prefill_is_opt_in_and_only_on_the_retry():
    item = {"messages": [{"role": "user", "content": "A는 8, B는 4이다. A÷B는?"}]}

    class Seen:
        def __init__(self):
            self.batches = []

        def generate(self, batch):
            self.batches.append([x.get("prefill") for x in batch])
            return [("정답: 2" if len(self.batches) == 1 else "계산식: 8 / 4", 1)] * len(batch)

    on = Seen()
    out = CalcPipeline(on, prefill=True).generate([item])
    assert on.batches == [[None], ["계산식: "]] and out[0][0].endswith("정답: 2")
    off = Seen()
    CalcPipeline(off).generate([item])
    assert off.batches == [[None], [None]]


def test_ungrounded_accepts_percent_forms_of_question_numbers():
    from genesis.harness.gorilla import ungrounded
    question = "원금 480만원, 연 3% 단리로 4년"
    assert ungrounded(question, "4800000 * 0.03 * 4") == [4800000.0]  # 480 is present, 4800000 is not
    assert ungrounded(question, "480 * 0.03 * 4") == []
    assert ungrounded("정가 200원, 할인율 0.15", "200 * 15 / 100") == []


def test_direction_flag_catches_dividing_by_the_new_value():
    from genesis.harness.gorilla import direction_flag
    question = "3,320에서 7,240으로 바뀌었다. 증감률은 몇 %인가?"
    assert direction_flag(question, "(7240 - 3320) / 7240 * 100") is True
    assert direction_flag(question, "(7240 - 3320) / 3320 * 100") is False
    assert direction_flag("평균은?", "(7240 - 3320) / 7240") is False  # not a rate question


def test_pipeline_counts_direction_flags_without_changing_the_answer():
    from genesis.harness.gorilla import CalcPipeline

    class Fake:
        def generate(self, batch):
            return [("계산식: (7240 - 3320) / 7240 * 100", 3)]

    item = {"messages": [{"role": "user", "content": "3320에서 7240으로. 증감률은?"}]}
    pipeline = CalcPipeline(Fake())
    out = pipeline.generate([item])
    assert pipeline.direction == 1 and out[0][0].endswith("정답: 54.14")


def test_direction_retry_is_opt_in_and_only_replaces_with_a_fixed_formula():
    from genesis.harness.gorilla import CalcPipeline

    class Fake:
        def __init__(self):
            self.prompts = []

        def generate(self, batch):
            self.prompts.append([b["messages"][-1]["content"] for b in batch])
            if len(self.prompts) == 1:
                return [("계산식: (7240 - 3320) / 7240 * 100", 3)]
            return [("계산식: (7240 - 3320) / 3320 * 100", 3)]

    item = {"messages": [{"role": "user", "content": "3320에서 7240으로. 증감률은?"}]}
    off = CalcPipeline(Fake())
    assert off.generate([item])[0][0].endswith("정답: 54.14") and off.retried == 0
    fake = Fake()
    on = CalcPipeline(fake, direction_retry=True)
    out = on.generate([item])
    assert out[0][0].endswith("정답: 118.07") and on.direction_rescued == 1
    assert "처음 값" in fake.prompts[1][0]  # the hint reached the retry only
    assert "처음 값" not in fake.prompts[0][0]


def test_direction_flag_skips_decrease_rate_questions():
    from genesis.harness.gorilla import direction_flag
    assert direction_flag("7240에서 3320으로. 감소율은?", "(7240 - 3320) / 7240 * 100") is False


def test_unit_hint_is_opt_in_and_reaches_first_pass_and_retry():
    from genesis.harness.gorilla import CalcPipeline, UNIT_HINT

    class Fake:
        def __init__(self):
            self.prompts = []

        def generate(self, batch):
            self.prompts += [b["messages"][-1]["content"] for b in batch]
            return [("정답: 99", 2) for _ in batch]  # a number worked out in its head: the retry runs

    item = {"messages": [{"role": "user", "content": "q 몇 %인가?"}]}
    off = Fake(); CalcPipeline(off).generate([item])
    assert all(UNIT_HINT not in p for p in off.prompts)
    on = Fake(); CalcPipeline(on, unit_hint=True).generate([item])
    assert len(on.prompts) == 2 and all(UNIT_HINT in p for p in on.prompts)


def test_contradiction_finds_numbers_listed_as_excluded_but_used():
    from genesis.harness.gorilla import contradiction
    reply = "제외: 4\n계산식: 2000*(1+0.04)^5 - 4"
    assert contradiction(reply, "2000*(1+0.04)^5 - 4") == [4.0]
    assert contradiction("제외: 4\n계산식: 2000*(1+0.04)^5", "2000*(1+0.04)^5") == []
    assert contradiction("제외: 없음\n계산식: 3 + 4", "3 + 4") == []
    assert contradiction("계산식: 3 + 4", "3 + 4") == []


def test_exclude_check_retries_a_self_contradicting_formula_and_keeps_the_fixed_one():
    from genesis.harness.gorilla import CalcPipeline

    class Fake:
        def __init__(self):
            self.prompts = []

        def generate(self, batch):
            self.prompts.append([b["messages"][-1]["content"] for b in batch])
            if len(self.prompts) == 1:
                return [("제외: 4\n계산식: 100 - 4", 5)]
            return [("계산식: 100", 3)]

    item = {"messages": [{"role": "user", "content": "금액 100, 수수료 4는 별도. 금액은?"}]}
    off = CalcPipeline(Fake())
    assert off.generate([item])[0][0].endswith("정답: 96") and off.retried == 0
    fake = Fake()
    on = CalcPipeline(fake, exclude_check=True)
    out = on.generate([item])
    assert out[0][0].endswith("정답: 100") and on.contradiction_rescued == 1 and on.contradicted == 1
    assert "제외한 숫자" in fake.prompts[1][0] and "제외:" in fake.prompts[0][0]
