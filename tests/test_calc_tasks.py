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
