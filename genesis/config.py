"""Reproducible architecture configuration and validation."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 257  # 256 UTF-8 byte values + EOS
    context_length: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None     # grouped-query attention; None = n_heads
    d_model: int = 96
    n_experts: int = 4                # 1 = dense SwiGLU MLP (no router)
    experts_per_token: int = 2
    expert_hidden_dim: int = 192      # SwiGLU hidden width (per expert)
    dropout: float = 0.0
    balance_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    position: str = "rope"            # rope | learned
    rope_theta: float = 10000.0
    norm: str = "rmsnorm"             # rmsnorm | layernorm
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    attention_bias: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or self.context_length <= 1:
            raise ValueError("vocab_size and context_length must be > 1")
        if min(self.n_layers, self.n_heads, self.d_model, self.n_experts, self.expert_hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.kv_heads <= 0 or self.n_heads % self.kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("experts_per_token must be in [1, n_experts]")
        if not 0 <= self.dropout < 1 or self.balance_loss_weight < 0 or self.router_z_loss_weight < 0:
            raise ValueError("invalid dropout or auxiliary loss weight")
        if self.position not in {"rope", "learned"}:
            raise ValueError("position must be 'rope' or 'learned'")
        if self.position == "rope" and self.head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        if self.norm not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm must be 'rmsnorm' or 'layernorm'")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = {k for k in data if not k.startswith("_")} - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
