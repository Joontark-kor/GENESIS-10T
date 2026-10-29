import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from genesis.config import ModelConfig
from genesis.data.shards import TokenSplit
from genesis.generate import load_model
from genesis.prepare import prepare
from genesis.train import evaluate, lr_at, main as train_main
from genesis.tokenizer import ByteTokenizer

TEXT = ("한국어 언어모델은 다음 토큰을 예측하도록 학습합니다. 데이터의 품질과 출처가 중요합니다. "
        "작은 실험으로 수치가 맞는지 먼저 확인합니다. ")


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    words = TEXT.split()
    rng = np.random.default_rng(0)
    for i in range(30):  # distinct documents so near-dedup keeps them
        body = " ".join(rng.choice(words, size=60))
        (raw / f"doc{i}.txt").write_text(f"문서 {i}. {body}", encoding="utf-8")
    out = tmp_path / "data"
    prepare(raw, out, source="unit", license_id="self-authored", val_fraction=0.2, test_fraction=0.2)
    return out


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps({"vocab_size": 257, "context_length": 32, "n_layers": 2, "n_heads": 4,
                                "d_model": 32, "n_experts": 2, "experts_per_token": 1,
                                "expert_hidden_dim": 64}), encoding="utf-8")
    return path


def _losses(run_dir: Path) -> list[float]:
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [r["train_loss"] for r in lines if r["type"] == "train"]


def _args(config_path, dataset, out, steps, *extra):
    return ["--config", str(config_path), "--data", str(dataset), "--out-dir", str(out),
            "--steps", str(steps), "--batch-size", "4", "--grad-accum", "2", "--eval-every", "3",
            "--device", "cpu", "--lr", "3e-3", "--warmup-steps", "2", *extra]


def test_lr_schedule_warmup_then_cosine():
    assert lr_at(1, 1.0, 10, 100, 0.1) == pytest.approx(0.1)
    assert lr_at(10, 1.0, 10, 100, 0.1) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 10, 100, 0.1) == pytest.approx(0.1)
    assert 0.1 < lr_at(55, 1.0, 10, 100, 0.1) < 1.0


def test_training_learns_and_resume_is_exact(tmp_path, dataset, config_path):
    torch.use_deterministic_algorithms(True)
    try:
        straight = tmp_path / "straight"
        result = train_main(_args(config_path, dataset, straight, 8))
        losses = _losses(straight)
        assert len(losses) == 8 and losses[-1] < losses[0]
        assert result["val"]["bpb"] > 0 and math.isfinite(result["val"]["loss"])

        resumed = tmp_path / "resumed"
        train_main(_args(config_path, dataset, resumed, 8, "--stop-after", "4"))
        # continue the same run to 8 steps: identical batches and optimizer state
        train_main(_args(config_path, dataset, resumed, 8, "--resume"))
        assert np.allclose(_losses(resumed), losses, rtol=0, atol=1e-6)
    finally:
        torch.use_deterministic_algorithms(False)


def test_resume_rejects_different_config(tmp_path, dataset, config_path):
    out = tmp_path / "run"
    train_main(_args(config_path, dataset, out, 2))
    other = tmp_path / "other.json"
    cfg = json.loads(config_path.read_text())
    cfg["d_model"] = 64
    other.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        train_main(_args(other, dataset, out, 4, "--resume"))


def test_vocab_mismatch_is_rejected(tmp_path, dataset):
    small_vocab = tmp_path / "v.json"
    small_vocab.write_text(json.dumps({"vocab_size": 100, "context_length": 16, "n_layers": 1,
                                       "n_heads": 2, "d_model": 16, "n_experts": 1,
                                       "experts_per_token": 1, "expert_hidden_dim": 32}))
    with pytest.raises(ValueError):
        train_main(["--config", str(small_vocab), "--data", str(dataset), "--out-dir",
                    str(tmp_path / "x"), "--steps", "1", "--device", "cpu"])


def test_bpb_equals_loss_over_ln2_for_bytes(dataset, config_path):
    """With a byte tokenizer, bits-per-byte is loss/ln2 rescaled for EOS tokens (0 bytes)."""
    from genesis.model import GenesisLM
    torch.manual_seed(0)
    model = GenesisLM(ModelConfig.from_json(config_path))
    split = TokenSplit(dataset, "val")
    tok = ByteTokenizer()
    result = evaluate(model, split, 32, 4, torch.device("cpu"), tok.token_byte_lengths(), None,
                      torch.autocast("cpu", enabled=False))
    assert result["bytes"] <= result["tokens"]
    expected = result["loss"] * result["tokens"] / math.log(2) / result["bytes"]
    assert result["bpb"] == pytest.approx(expected)
    # untrained model is close to uniform over 257 symbols
    assert abs(result["loss"] - math.log(257)) < 0.5


def test_generate_loads_checkpoint_with_tokenizer(tmp_path, dataset, config_path):
    out = tmp_path / "run"
    train_main(_args(config_path, dataset, out, 2))
    model, tokenizer, checkpoint = load_model(out / "model.pt")
    ids = model.generate(torch.tensor([tokenizer.encode("한국어")]), new_tokens=5, top_k=10)
    assert ids.shape[1] <= 3 * 3 + 5
    assert isinstance(tokenizer.decode(ids[0].tolist()), str)
    assert checkpoint["tokenizer"]["type"] == "byte"
