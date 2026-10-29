from dataclasses import asdict
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from genesis.config import ModelConfig
from genesis.export_hf import export, verify_export
from genesis.model import GenesisLM
from genesis.tokenizer import train_bpe

CORPUS = ["대한민국의 수도는 서울이다.", "한국어 언어모델을 처음부터 학습한다.",
          "Hello world 123", "토크나이저와 모델을 함께 내보낸다."] * 40


def _checkpoint(tmp_path: Path, **overrides) -> Path:
    tok = train_bpe(iter(CORPUS), vocab_size=400, output=tmp_path / "tok.json", show_progress=False)
    base = dict(vocab_size=400, context_length=32, n_layers=2, n_heads=4, n_kv_heads=2, d_model=32,
                n_experts=1, experts_per_token=1, expert_hidden_dim=64)
    base.update(overrides)
    config = ModelConfig(**base)
    torch.manual_seed(0)
    model = GenesisLM(config)
    with torch.no_grad():  # non-trivial norm weights so a swapped norm would be caught
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": asdict(config), "tokenizer": tok.describe(),
                "step": 0, "parameters": 0}, path)
    return path


@pytest.mark.parametrize("overrides", [{}, {"tie_embeddings": False}, {"attention_bias": True},
                                       {"n_kv_heads": None}])
def test_export_matches_logits(tmp_path, overrides):
    ckpt = _checkpoint(tmp_path, **overrides)
    out = export(ckpt, tmp_path / "hf", verify=False)
    assert verify_export(ckpt, out) < 1e-4


def test_verify_catches_a_wrong_weight(tmp_path):
    ckpt = _checkpoint(tmp_path)
    out = export(ckpt, tmp_path / "hf", verify=False)
    from safetensors.torch import load_file, save_file
    weights = load_file(out / "model.safetensors")
    weights["model.layers.0.mlp.up_proj.weight"] = weights["model.layers.0.mlp.up_proj.weight"] * 1.5
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(AssertionError):
        verify_export(ckpt, out)


def test_moe_and_byte_checkpoints_are_rejected(tmp_path):
    ckpt = _checkpoint(tmp_path, n_experts=2, experts_per_token=1)
    with pytest.raises(ValueError):
        export(ckpt, tmp_path / "hf", verify=False)
