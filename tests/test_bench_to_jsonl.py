import json

import pytest

from genesis.bench_to_jsonl import PRESETS, export_benchmark, item_text


def fake_loader(path, name, split):
    fields = {f for entries in PRESETS.values() for (p, n, s, fs) in entries if p == path and n == name for f in fs}
    return [{f: (["가", "나"] if f == "choices" else f"{name}-{split}-{f}-{i}") for f in fields} for i in range(3)]


@pytest.mark.parametrize("benchmark", sorted(PRESETS))
def test_export_writes_one_item_per_record(tmp_path, benchmark):
    summary = export_benchmark(benchmark, tmp_path, loader=fake_loader)
    lines = (tmp_path / f"{benchmark}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["items"] > 0
    first = json.loads(lines[0])
    assert first["benchmark"] == benchmark and first["text"]


def test_item_text_flattens_choices():
    assert item_text({"q": "질문", "choices": ["A", "B"], "empty": ""}, ["q", "choices", "empty"]) == "질문\nA\nB"
