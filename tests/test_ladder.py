import json
from pathlib import Path

import numpy as np
import pytest

from genesis.ladder import fit_power_law, plan_runs, predict, report, write_plan

ROOT = Path(__file__).parent.parent


def _manifest(tmp_path: Path, train_tokens: int) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"format_version": 2, "splits": {
        "train": {"tokens": train_tokens}, "val": {"tokens": 1}, "test": {"tokens": 1}}}))
    return data


def test_plan_caps_epochs_and_writes_script(tmp_path):
    data = _manifest(tmp_path, train_tokens=500_000_000)
    configs = [ROOT / "configs/dense-40m.json", ROOT / "configs/dense-300m.json"]
    plan = plan_runs(data, configs, out_root=tmp_path / "ladder")
    small, big = plan["runs"]
    assert small["tokens"] == pytest.approx(20 * small["parameters"], rel=0.01)   # 20 tokens/param
    assert big["epochs"] == pytest.approx(4.0, rel=0.01)                          # capped at 4 epochs
    script = write_plan(plan, tmp_path / "ladder")
    text = script.read_text()
    assert "--resume" in text and "dense-300m" in text


def test_power_law_fit_recovers_parameters():
    sizes = [2e7, 7e7, 2.5e8, 9e8]
    true = {"E": 0.6, "A": 40.0, "alpha": 0.3}
    losses = [predict(true, n) for n in sizes]
    fit = fit_power_law(sizes, losses)
    assert fit["alpha"] == pytest.approx(0.3, rel=0.05)
    assert predict(fit, 3e9) == pytest.approx(predict(true, 3e9), rel=0.01)


def _fake_run(root: Path, name: str, n: int, bpb: float, moe: bool = False):
    run = root / name
    run.mkdir(parents=True)
    run.joinpath("run.json").write_text(json.dumps({
        "config": {"n_experts": 8 if moe else 1}, "parameters": n * 2,
        "non_embedding_active_parameters": n, "tokens_per_step": 1000, "steps": 10}))
    run.joinpath("metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "train", "step": 10, "tokens_per_s": 5000.0},
        {"type": "eval", "step": 10, "loss": bpb * 3, "bpb": bpb}]))


def test_report_fits_dense_and_lists_moe(tmp_path):
    root = tmp_path / "runs"
    for name, n in [("a", 2e7), ("b", 7e7), ("c", 2.5e8)]:
        _fake_run(root, name, int(n), 0.6 + 40 * n ** -0.3)
    _fake_run(root, "moe", int(7e7), 0.6 + 40 * 7e7 ** -0.3 - 0.02, moe=True)
    result = report(root, [ROOT / "configs/dense-1b.json"], plot=tmp_path / "plot.png")
    assert result["fit"]["alpha"] == pytest.approx(0.3, rel=0.1)
    moe = next(r for r in result["runs"] if r["name"] == "moe")
    assert moe["val_bpb"] < moe["dense_fit_bpb_at_same_N"]
    assert result["predictions"][0]["predicted_val_bpb"] < min(r["val_bpb"] for r in result["runs"])
    assert (tmp_path / "plot.png").exists()
