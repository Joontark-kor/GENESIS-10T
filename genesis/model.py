"""Small executable causal MoE Transformer, built from PyTorch primitives.

This is a correctness baseline; routing is a naive local implementation and
DOES NOT provide distributed expert parallelism or high-performance kernels.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.proj(attended)


class ExpertMLP(nn.Module):
    """SwiGLU expert; params per expert approximately 3*d_model*hidden_dim."""
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(width, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.k = config.experts_per_token
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertMLP(config.d_model, config.expert_hidden_dim)
            for _ in range(config.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        routing_logits = self.router(flat).float()
        top_values, top_ids = torch.topk(routing_logits, k=self.k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1).to(flat.dtype)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_indices, top_positions = torch.where(top_ids == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_input = flat.index_select(0, token_indices)
            expert_output = expert(expert_input)
            contribution = expert_output * top_weights[token_indices, top_positions].unsqueeze(-1)
            result.index_add_(0, token_indices, contribution)

        # Basic auxiliary load-balancing loss. This is a research baseline,
        # not a full production-grade, capacity-limited expert router.
        probabilities = F.softmax(routing_logits, dim=-1)
        first_choice_load = F.one_hot(top_ids[:, 0], num_classes=self.n_experts).float().mean(dim=0)
        balance_loss = self.n_experts * (probabilities.mean(dim=0) * first_choice_load.detach()).sum()
        return result.reshape(original_shape), balance_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.moe_norm = nn.LayerNorm(config.d_model)
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        moe_output, balance_loss = self.moe(self.moe_norm(x))
        return x + self.dropout(moe_output), balance_loss


class GenesisLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if ids.ndim != 2:
            raise ValueError("expected [batch, sequence] token ids")
        batch, seq_len = ids.shape
        if seq_len > self.config.context_length:
            raise ValueError("sequence longer than configured context_length")
        positions = torch.arange(seq_len, device=ids.device)
        hidden = self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
        aux_losses = []
        for block in self.blocks:
            hidden, aux = block(hidden)
            aux_losses.append(aux)
        logits = self.lm_head(self.final_norm(hidden))
        aux_loss = torch.stack(aux_losses).mean()
        result: dict[str, torch.Tensor] = {"logits": logits, "balance_loss": aux_loss}
        if targets is not None:
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            result["lm_loss"] = lm_loss
            result["loss"] = lm_loss + self.config.balance_loss_weight * aux_loss
        return result

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, new_tokens: int = 40, temperature: float = 1.0,
                 top_k: int | None = None, eos_id: int = 256) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if new_tokens < 0 or ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("invalid prompt or new_tokens")
        self.eval()
        for _ in range(new_tokens):
            context = ids[:, -self.config.context_length:]
            logits = self(context)["logits"][:, -1, :] / temperature
            if top_k is not None:
                if not 1 <= top_k <= logits.shape[-1]:
                    raise ValueError("top_k out of range")
                threshold = torch.topk(logits, k=top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            if torch.all(next_id == eos_id):
                break
        return ids
