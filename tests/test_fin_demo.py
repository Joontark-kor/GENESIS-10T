from types import SimpleNamespace

from genesis_fin.dart import normalize
from genesis_fin.demo import answer, find_in_records, parse_question, render
from tests.test_fin_tasks import SAMSUNG_LIKE


def record(year=2024):
    rec = normalize(SAMSUNG_LIKE)
    rec.update({"corp_code": "00126380", "corp_name": "삼성전자", "year": year, "fs_div": "CFS", "report": "annual"})
    return rec


def finder(records):
    return lambda squashed, year: find_in_records(records, squashed, year)


def fake(answer_text, formula="92228 / 363678 * 100", warnings=()):
    return lambda messages: SimpleNamespace(answer=answer_text, formula=formula, warnings=list(warnings))


def test_parse_question_reads_metric_and_year():
    assert parse_question("삼성전자 2024년 부채 비율은?")[:2] == ("debt_ratio", 2024)
    assert parse_question("삼성전자 2024년 자기자본비율은?")[0] == "equity_ratio"
    assert parse_question("삼성전자 매출 증가율은?")[:2] == ("revenue_growth", None)
    assert parse_question("삼성전자는 어떤 회사야?")[0] is None


def test_correct_answer_passes_the_check():
    out = answer("삼성전자 2024년 부채비율은?", finder([record()]), fake("25.36%"))
    assert out.ok and out.check == "pass" and abs(out.expected - 25.36) < 1e-9
    assert "검산: 통과" in render(out) and "출처: OpenDART 2024년" in render(out)


def test_wrong_answer_is_flagged_not_hidden():
    out = answer("삼성전자 2024년 부채비율은?", finder([record()]), fake("52.36%"))
    assert out.check == "fail"
    assert "불일치" in render(out) and "쓰지 마세요" in render(out)


def test_no_value_and_warnings_are_shown():
    out = answer("삼성전자 2024년 영업이익률은?", finder([record()]), fake(None, None, ["식 없이 답했습니다(계산기를 쓰지 않음)"]))
    assert out.check == "no value"
    text = render(out)
    assert "검산: 할 수 없음" in text and "확인할 점: 식 없이" in text


def test_out_of_scope_and_missing_filing_never_reach_the_model():
    def boom(messages):
        raise AssertionError("the model must not be called")

    assert not answer("삼성전자는 어떤 회사야?", finder([record()]), boom).ok
    assert "범위 밖" in render(answer("삼성전자는 어떤 회사야?", finder([record()]), boom))
    assert not answer("카카오 2024년 부채비율은?", finder([record()]), boom).ok
    assert "공시를 찾지 못했습니다" in render(answer("삼성전자 2020년 부채비율은?", finder([record()]), boom))


def test_latest_year_is_used_when_the_question_has_none():
    out = answer("삼성전자 부채비율은?", finder([record(2023), record(2024)]), fake("25.36%"))
    assert out.year == 2024
