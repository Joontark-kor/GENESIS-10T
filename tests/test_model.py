import pytest
import torch

from genesis.config import ModelConfig
from genesis.model import GenesisLM

VARIANTS = {
    "moe-rope-rms": dict(n_experts=3, experts_per_token=2),
    "dense": dict(n_experts=1, experts_per_token=1),
    "gqa": dict(n_heads=4, n_kv_heads=2, n_experts=1, experts_per_token=1),
    "legacy": dict(position="learned", norm="layernorm", attention_bias=True, n_experts=2, experts_per_token=1),
    "untied": dict(tie_embeddings=False, n_experts=1, experts_per_token=1),
}


def small(**overrides) -> ModelConfig:
    base = dict(context_length=16, n_layers=2, d_model=32, n_heads=4, expert_hidden_dim=64)
    base.update(overrides)
    return ModelConfig(**base)


def test_config_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        ModelConfig(d_model=97, n_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(n_experts=1, experts_per_token=2)
    with pytest.raises(ValueError):
        ModelConfig.from_dict({"d_model": 96, "typo_field": 1})


@pytest.mark.parametrize("name", VARIANTS)
def test_forward_backward(name):
    torch.manual_seed(12)
    model = GenesisLM(small(**VARIANTS[name]))
    x = torch.randint(0, 257, (2, 12))
    out = model(x, x)
    assert out["logits"].shape == (2, 12, 257)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.blocks[0].attention.qkv.weight.grad is not None
    if model.config.is_moe:
        assert model.blocks[0].ffn.router.weight.grad is not None
        assert out["expert_load"].shape == (2, model.config.n_experts)
        assert torch.allclose(out["expert_load"].sum(-1), torch.ones(2))
    else:
        assert float(out["balance_loss"]) == 0.0


@pytest.mark.parametrize("name", VARIANTS)
def test_strict_causal_mask(name):
    torch.manual_seed(7)
    model = GenesisLM(small(**VARIANTS[name])).eval()
    a = torch.tensor([[3, 4, 5, 6, 7, 8]])
    b = torch.tensor([[3, 4, 5, 91, 92, 93]])
    with torch.no_grad():
        pa = model(a)["logits"][:, :3]
        pb = model(b)["logits"][:, :3]
    assert torch.allclose(pa, pb, atol=1e-5)


def test_rope_preserves_norm():
    torch.manual_seed(3)
    model = GenesisLM(small(n_experts=1, experts_per_token=1, n_layers=1)).eval()
    q = model.blocks[0].attention
    x = torch.randn(1, 4, 6, 8)
    rq, rk = q.rope(x, x)
    assert torch.allclose(rq.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # rotation preserves norm


def test_balance_loss_is_one_when_uniform():
    torch.manual_seed(0)
    model = GenesisLM(small(n_experts=4, experts_per_token=2))
    moe = model.blocks[0].ffn
    torch.nn.init.zeros_(moe.router.weight)  # uniform router probabilities
    _, aux = moe(torch.randn(64, 32))
    assert aux["balance"].item() == pytest.approx(1.0, abs=1e-5)


def test_generate_respects_context_and_eos():
    model = GenesisLM(small(context_length=8, n_layers=1, n_experts=2, experts_per_token=1))
    prompt = torch.randint(0, 255, (1, 12))
    with torch.no_grad():
        generated = model.generate(prompt, new_tokens=3, top_k=4)
    assert generated.shape == (1, 15)
