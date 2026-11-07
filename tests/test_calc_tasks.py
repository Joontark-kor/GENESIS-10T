import random

from genesis.harness.scorers import parse_number
from genesis_calc.build_tasks import KINDS, build, make


def test_every_kind_has_a_finite_gold_and_unique_questions(tmp_path):
    manifest = build(tmp_path, per_kind=20, seed=3)
    assert manifest["items"] == 20 * len(KINDS)
    import json
    rows = [json.loads(x) for x in (tmp_path / "eval_test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["messages"][1]["content"] for r in rows}) == len(rows)
    assert all(abs(r["gold"]) < 1e9 for r in rows)


def test_reference_answer_parses_back_to_gold():
    rng = random.Random(0)
    for kind in KINDS:
        _, gold, tol = make(kind, rng)
        assert abs(parse_number(f"정답: {gold:g}") - gold) <= tol


def test_same_seed_same_items(tmp_path):
    build(tmp_path / "a", 5, 7); build(tmp_path / "b", 5, 7)
    assert (tmp_path / "a/eval_test.jsonl").read_text(encoding="utf-8") == (tmp_path / "b/eval_test.jsonl").read_text(encoding="utf-8")


def test_hard_set_has_finite_golds_and_reference_parses_back(tmp_path):
    import json
    from genesis_calc.build_tasks import HARD_KINDS, make_hard
    manifest = build(tmp_path, per_kind=15, seed=2, kinds_set="hard")
    assert manifest["items"] == 15 * len(HARD_KINDS) and manifest["set"] == "hard"
    rows = [json.loads(x) for x in (tmp_path / "eval_test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["messages"][1]["content"] for r in rows}) == len(rows)
    rng = random.Random(0)
    for kind in HARD_KINDS:
        for _ in range(20):
            _, gold, tol = make_hard(kind, rng)
            assert 0 < gold < 1e6 and abs(parse_number(f"정답: {gold:g}") - gold) <= tol


def test_caret_is_read_as_a_power_not_rejected():
    from genesis.tools import calculate
    assert calculate("2200 * (1 + 0.045)^2 - 3") == __import__("pytest").approx(2402.45, abs=0.01)
    assert calculate("(5829/3100)^(1/2)-1") == __import__("pytest").approx(0.3713, abs=1e-3)
    assert calculate("2^3") == 8
