"""Paired comparison, seed summaries and resumable evaluation runs (no model needed)."""
import json
import math

import pytest

from genesis.harness import evaluate as ev


def _report(name, correct, kinds=None):
    rows = [{"id": f"q{i}", "kind": (kinds or ["a"] * len(correct))[i], "correct": c, "prediction": 1 if c else None,
             "gold": i} for i, c in enumerate(correct)]
    return {"name": name, "items": len(rows), **ev.summarize(rows), "rows": rows}


def test_paired_counts_and_mcnemar_z():
    first = _report("base", [True] * 6 + [False] * 4)
    second = _report("pb", [True] * 2 + [False] * 4 + [True] * 2 + [False] * 2)
    result = ev.paired(first, second)
    assert (result["both"], result["first_only"], result["second_only"], result["neither"]) == (2, 4, 2, 2)
    assert result["z"] == round((4 - 2) / math.sqrt(6), 2)
    # swapping the reports flips the sign
    assert ev.paired(second, first)["z"] == -result["z"]


def test_paired_reproduces_the_gsm8k_numbers_from_the_docs():
    # 143 vs 85 only-one-right gave z=3.84 in docs/12_polarbear_ko.md
    first = _report("base", [True] * 143 + [False] * 85)
    second = _report("pb", [False] * 143 + [True] * 85)
    assert ev.paired(first, second)["z"] == 3.84


def test_paired_refuses_different_items():
    first = _report("a", [True, False])
    second = _report("b", [True, False, True])
    with pytest.raises(ValueError):
        ev.paired(first, second)
    assert "not comparable" in ev.paired_markdown([first, second])


def test_seed_summary_mean_and_range(tmp_path):
    reports = [_report(f"s{i}", [True] * k + [False] * (10 - k)) for i, k in enumerate([7, 8, 9])]
    (summary,) = ev.seed_summary({"pb": reports})
    assert summary["runs"] == 3 and summary["min"] == 0.7 and summary["max"] == 0.9
    assert summary["mean"] == pytest.approx(0.8)
    assert "70.0%-90.0%" in ev.seeds_markdown([summary])


def test_seeds_command_reads_globbed_reports(tmp_path, capsys):
    for seed, k in enumerate([6, 8], start=1):
        (tmp_path / f"r-s{seed}-pb.json").write_text(json.dumps(_report(f"s{seed}", [True] * k + [False] * (10 - k))))
    ev.main(["seeds", f"pb={tmp_path}/r-s*-pb.json"])
    out = capsys.readouterr().out
    assert "| pb | 2 | 70.0% | 60.0%-80.0% |" in out


class _Fake:
    """Answers the gold number; dies after `die_after` n_calls to simulate a lost session."""
    hardware = "fake"
    die_after = None
    n_calls = 0

    def __init__(self, args):
        pass

    def sync(self):
        pass

    def generate(self, batch):
        type(self).n_calls += 1
        if type(self).die_after is not None and type(self).n_calls > type(self).die_after:
            raise RuntimeError("session lost")
        return [(f"답: {item['gold']}", 3) for item in batch]


def _data(tmp_path, n=10):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"q{i}", "kind": "a", "gold": i,
                                          "messages": [{"role": "user", "content": f"{i}?"}]})
                             for i in range(n)), encoding="utf-8")
    return path


def _run(tmp_path, data, out, *extra):
    return ev.main(["run", "--endpoint", "http://x/v1", "--model", "m", "--data", str(data), "--name", "t",
                    "--batch-size", "2", "--out", str(out), *extra])


def test_resume_continues_after_a_crash_and_matches_a_clean_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EndpointGenerator", _Fake)
    data = _data(tmp_path)
    out = tmp_path / "r.json"
    _Fake.n_calls, _Fake.die_after = 0, 3
    with pytest.raises(RuntimeError):
        _run(tmp_path, data, out)
    partial = tmp_path / "r.json.partial.jsonl"
    assert partial.exists() and not out.exists()
    assert sum(len(json.loads(line)["rows"]) for line in partial.read_text().splitlines()) == 6
    _Fake.n_calls, _Fake.die_after = 0, None
    _run(tmp_path, data, out, "--resume")
    assert _Fake.n_calls == 2, "only the 4 missing items are generated again"
    resumed = json.loads(out.read_text())
    assert not partial.exists()
    clean = _run(tmp_path, data, tmp_path / "clean.json")
    assert [r["id"] for r in resumed["rows"]] == [r["id"] for r in clean["rows"]]
    assert resumed["accuracy"] == clean["accuracy"] == 1.0
    assert resumed["speed"]["generated_tokens"] == clean["speed"]["generated_tokens"] == 30


def test_resume_starts_over_when_the_data_changed_and_without_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EndpointGenerator", _Fake)
    out = tmp_path / "r.json"
    _Fake.n_calls, _Fake.die_after = 0, 2
    with pytest.raises(RuntimeError):
        _run(tmp_path, _data(tmp_path), out)
    _Fake.n_calls, _Fake.die_after = 0, None
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(json.dumps({"id": f"q{i}", "kind": "a", "gold": i + 100,
                                           "messages": [{"role": "user", "content": "?"}]}) for i in range(10)))
    _run(tmp_path, other, out, "--resume")
    assert _Fake.n_calls == 5  # different gold answers: nothing reused
    # without --resume an old partial file is discarded, never mixed in
    (tmp_path / "r.json.partial.jsonl").write_text('{"rows": []}\n')
    _Fake.n_calls = 0
    _run(tmp_path, other, out)
    assert _Fake.n_calls == 5


def test_resume_needs_out(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ev.main(["run", "--endpoint", "http://x/v1", "--model", "m", "--data", str(_data(tmp_path)), "--name", "t",
                 "--resume"])
