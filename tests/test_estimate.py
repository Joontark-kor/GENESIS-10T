import pytest

from genesis.config import ModelConfig
from genesis.estimate import estimate, flops_per_token
from genesis.model import GenesisLM
from tests.test_model import VARIANTS, small


@pytest.mark.parametrize("name", VARIANTS)
def test_estimate_matches_real_module(name):
    cfg = small(**VARIANTS[name])
    actual = sum(p.numel() for p in GenesisLM(cfg).parameters())
    assert estimate(cfg)["total_parameters"] == actual


def test_debug_and_shipped_configs_match(tmp_path):
    from pathlib import Path
    for path in sorted((Path(__file__).parent.parent / "configs").glob("*.json")):
        if path.name.startswith("sources"):
            continue
        cfg = ModelConfig.from_json(path)
        assert estimate(cfg)["total_parameters"] > 0
        if estimate(cfg)["total_parameters"] < 50_000_000:
            assert estimate(cfg)["total_parameters"] == sum(p.numel() for p in GenesisLM(cfg).parameters())


def test_hypothetical_10t_sizing():
    cfg = ModelConfig(vocab_size=131072, context_length=8192, n_layers=96,
                      n_heads=64, d_model=8192, n_experts=256,
                      experts_per_token=2, expert_hidden_dim=16384)
    est = estimate(cfg)
    assert 9.9e12 < est["total_parameters"] < 10.1e12
    assert 100e9 < est["approx_active_parameters_per_token"] < 120e9
    assert flops_per_token(cfg) > 6 * 100e9
