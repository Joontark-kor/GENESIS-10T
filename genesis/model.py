"""Decoder-only Transformer with optional sparse Mixture-of-Experts FFN.

Llama-style defaults: RMSNorm pre-normalization, rotary position embeddings,
grouped-query attention through ``scaled_dot_product_attention`` (which picks a
FlashAttention/memory-efficient kernel when available), SwiGLU MLP.

``n_experts == 1`` gives the dense baseline; ``n_experts > 1`` swaps the MLP for
a top-k routed MoE with a Switch-style load-balancing loss and a router z-loss.
Routing here is a simple single-device loop: correct and readable, not an
expert-parallel implementation.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def build_norm(config: ModelConfig) -> nn.Module:
    if config.norm == "rmsnorm":
        return RMSNorm(config.d_model, config.norm_eps)
    return nn.LayerNorm(config.d_model, eps=config.norm_eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos[:seq_len].to(q.dtype)
        sin = self.sin[:seq_len].to(q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.n_heads, config.kv_heads, config.head_dim
        q_width, kv_width = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.qkv = nn.Linear(config.d_model, q_width + 2 * kv_width, bias=config.attention_bias)
        self.proj = nn.Linear(q_width, config.d_model, bias=config.attention_bias)
        self.split = (q_width, kv_width, kv_width)
        self.dropout = config.dropout
        self.rope = (RotaryEmbedding(self.head_dim, config.context_length, config.rope_theta)
                     if config.position == "rope" else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv(x).split(self.split, dim=-1)
        q = q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, seq_len, -1))


class ExpertMLP(nn.Module):
    """SwiGLU MLP; ~3*d_model*hidden parameters."""

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
        self.experts = nn.ModuleList(
            [ExpertMLP(config.d_model, config.expert_hidden_dim) for _ in range(config.n_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        logits = self.router(flat).float()
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=self.k, dim=-1)
        top_weights = (top_probs / top_probs.sum(dim=-1, keepdim=True)).to(flat.dtype)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot = torch.where(top_ids == expert_id)
            if token_idx.numel() == 0:
                continue
            out = expert(flat.index_select(0, token_idx))
            result.index_add_(0, token_idx, out * top_weights[token_idx, slot].unsqueeze(-1))
        # Switch-Transformer balance loss over all k assignments (=1 when uniform),
        # plus ST-MoE router z-loss that keeps router logits small and stable.
        load = F.one_hot(top_ids, self.n_experts).float().sum(dim=1).mean(dim=0) / self.k
        balance = self.n_experts * (probs.mean(dim=0) * load).sum()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return result.reshape(original_shape), {"balance": balance, "z_loss": z_loss, "load": load.detach()}


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = build_norm(config)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = build_norm(config)
        self.ffn = SparseMoE(config) if config.is_moe else ExpertMLP(config.d_model, config.expert_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        if isinstance(self.ffn, SparseMoE):
            out, aux = self.ffn(self.ffn_norm(x))
        else:
            out, aux = self.ffn(self.ffn_norm(x)), None
        return x + self.dropout(out), aux


class GenesisLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (nn.Embedding(config.context_length, config.d_model)
                                   if config.position == "learned" else None)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = build_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        # GPT-2 style: shrink residual-branch outputs so depth does not blow up activations.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("attention.proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

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
        _, seq_len = ids.shape
        if seq_len > self.config.context_length:
            raise ValueError("sequence longer than configured context_length")
        hidden = self.token_embedding(ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(torch.arange(seq_len, device=ids.device))[None]
        balances, z_losses, loads = [], [], []
        for block in self.blocks:
            hidden, aux = block(hidden)
            if aux is not None:
                balances.append(aux["balance"])
                z_losses.append(aux["z_loss"])
                loads.append(aux["load"])
        logits = self.lm_head(self.final_norm(hidden))
        zero = logits.new_zeros((), dtype=torch.float32)
        balance = torch.stack(balances).mean() if balances else zero
        z_loss = torch.stack(z_losses).mean() if z_losses else zero
        result: dict[str, torch.Tensor] = {"logits": logits, "balance_loss": balance, "z_loss": z_loss}
        if loads:
            result["expert_load"] = torch.stack(loads)
        if targets is not None:
            lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            result["lm_loss"] = lm_loss
            result["loss"] = (lm_loss + self.config.balance_loss_weight * balance
                              + self.config.router_z_loss_weight * z_loss)
        return result

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, new_tokens: int = 40, temperature: float = 1.0,
                 top_k: int | None = None, eos_id: int | None = None) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if new_tokens < 0 or ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("invalid prompt or new_tokens")
        self.eval()
        for _ in range(new_tokens):
            context = ids[:, -self.config.context_length:]
            logits = self(context)["logits"][:, -1, :].float() / temperature
            if top_k is not None:
                if not 1 <= top_k <= logits.shape[-1]:
                    raise ValueError("top_k out of range")
                threshold = torch.topk(logits, k=top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)
            if eos_id is not None and torch.all(next_id == eos_id):
                break
        return ids
