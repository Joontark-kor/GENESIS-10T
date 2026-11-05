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
    pipeline = CalcPipeline(Fake())
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
