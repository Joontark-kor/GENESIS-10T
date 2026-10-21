import torch
import pytest
from genesis.config import ModelConfig
from genesis.model import GenesisLM, SparseMoE
from genesis.tokenizer import ByteTokenizer


def test_byte_roundtrip():
    text = "안녕하세요. Hello, こんにちは 🌍"
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(text, add_eos=True)) == text


def test_config_rejects_invalid_heads():
    with pytest.raises(ValueError):
        ModelConfig(d_model=97, n_heads=4)


def test_forward_backward_and_moe_gradients():
    torch.manual_seed(12)
    model = GenesisLM(ModelConfig(context_length=16, n_layers=2, d_model=32,
                                  n_heads=4, n_experts=3, experts_per_token=2,
                                  expert_hidden_dim=64))
    x = torch.randint(0, 257, (2, 12))
    out = model(x, x)
    assert out["logits"].shape == (2, 12, 257)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.blocks[0].moe.router.weight.grad is not None
    assert model.blocks[0].attention.qkv.weight.grad is not None


def test_strict_causal_mask():
    torch.manual_seed(7)
    model = GenesisLM(ModelConfig(context_length=16, n_layers=1, d_model=32,
                                  n_heads=4, n_experts=2, experts_per_token=1,
                                  expert_hidden_dim=48)).eval()
    a = torch.tensor([[3, 4, 5, 6, 7, 8]])
    b = torch.tensor([[3, 4, 5, 91, 92, 93]])
    with torch.no_grad():
        pa = model(a)["logits"][:, :3]
        pb = model(b)["logits"][:, :3]
    assert torch.allclose(pa, pb, atol=1e-6)


def test_generate_respects_context():
    config = ModelConfig(context_length=8, n_layers=1, d_model=32,
                         n_heads=4, n_experts=2, experts_per_token=1,
                         expert_hidden_dim=48)
    model = GenesisLM(config)
    prompt = torch.randint(0, 255, (1, 12))
    with torch.no_grad():
        generated = model.generate(prompt, new_tokens=3, top_k=4)
    assert generated.shape == (1, 15)
