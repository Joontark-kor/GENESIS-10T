"""Reproducible architecture configuration and validation."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 257  # 256 UTF-8 byte values + EOS
    context_length: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 96
    n_experts: int = 4
    experts_per_token: int = 2
    expert_hidden_dim: int = 192
    dropout: float = 0.0
    balance_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or self.context_length <= 1:
            raise ValueError("vocab_size and context_length must be > 1")
        if min(self.n_layers, self.n_heads, self.d_model, self.n_experts, self.expert_hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("experts_per_token must be in [1, n_experts]")
        if not 0 <= self.dropout < 1 or self.balance_loss_weight < 0:
            raise ValueError("invalid dropout or balance loss weight")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
