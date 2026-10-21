from genesis.config import ModelConfig
from genesis.model import GenesisLM
from genesis.estimate import estimate


def test_estimate_matches_real_module():
    cfg = ModelConfig()
    actual = sum(p.numel() for p in GenesisLM(cfg).parameters())
    assert estimate(cfg)["total_parameters"] == actual


def test_hypothetical_10t_sizing():
    cfg = ModelConfig(vocab_size=131072, context_length=8192, n_layers=96,
                      n_heads=64, d_model=8192, n_experts=256,
                      experts_per_token=2, expert_hidden_dim=16384)
    est = estimate(cfg)
    assert 9.9e12 < est["total_parameters"] < 10.1e12
    assert 100e9 < est["approx_active_parameters_per_token"] < 120e9
