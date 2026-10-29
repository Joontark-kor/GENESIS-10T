"""Parameter, memory and compute estimates without allocating a model.

Counts correspond exactly to ``genesis.model.GenesisLM`` (a unit test checks
this), not to any particular industry model.
"""
from __future__ import annotations

import argparse

from .config import ModelConfig


def estimate(config: ModelConfig) -> dict[str, int]:
    d, layers, experts, hidden = config.d_model, config.n_layers, config.n_experts, config.expert_hidden_dim
    norm = d if config.norm == "rmsnorm" else 2 * d
    q_width, kv_width = config.n_heads * config.head_dim, config.kv_heads * config.head_dim
    attn = d * (q_width + 2 * kv_width) + q_width * d
    if config.attention_bias:
        attn += q_width + 2 * kv_width + d
    expert = 3 * d * hidden
    ffn = (d * experts + experts * expert) if config.is_moe else expert
    per_layer = 2 * norm + attn + ffn
    embedding = config.vocab_size * d
    positions = config.context_length * d if config.position == "learned" else 0
    head = 0 if config.tie_embeddings else config.vocab_size * d
    total = embedding + positions + layers * per_layer + norm + head
    inactive = layers * (experts - config.experts_per_token) * expert if config.is_moe else 0
    active = total - inactive
    return {
        "total_parameters": total,
        "approx_active_parameters_per_token": active,
        "non_embedding_active_parameters": active - embedding - positions,
        "bf16_weights_bytes": 2 * total,
        "adam_example_state_bytes": 16 * total,
    }


def flops_per_token(config: ModelConfig, seq_len: int | None = None) -> int:
    """Training FLOPs per token: 6 * active non-embedding params + attention scores.

    The lm_head matmul is included (it is a real matmul even when tied).
    """
    seq_len = seq_len or config.context_length
    counts = estimate(config)
    matmul_params = counts["non_embedding_active_parameters"] + config.vocab_size * config.d_model
    attention = 6 * config.n_layers * config.d_model * seq_len  # QK^T and AV, fwd+bwd (Kaplan et al.)
    return 6 * matmul_params + attention


def training_flops(config: ModelConfig, tokens: float) -> float:
    return flops_per_token(config) * tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.json")
    parser.add_argument("--hypothetical-10t", action="store_true")
    parser.add_argument("--tokens", type=float, help="training tokens, e.g. 2.5e9, to print compute")
    parser.add_argument("--gpu-tflops", type=float, default=989.0, help="peak dense BF16 TFLOPS per GPU")
    parser.add_argument("--mfu", type=float, default=0.4, help="assumed model FLOPs utilization")
    args = parser.parse_args()
    if args.hypothetical_10t:
        config = ModelConfig(vocab_size=131072, context_length=8192, n_layers=96,
                             n_heads=64, d_model=8192, n_experts=256,
                             experts_per_token=2, expert_hidden_dim=16384)
    else:
        config = ModelConfig.from_json(args.config)
    for key, value in estimate(config).items():
        print(f"{key}: {value:,}")
    if args.tokens:
        total = training_flops(config, args.tokens)
        gpu_hours = total / (args.gpu_tflops * 1e12 * args.mfu) / 3600
        print(f"training_flops: {total:.3e} for {args.tokens:.3e} tokens")
        print(f"gpu_hours: {gpu_hours:,.1f} at {args.gpu_tflops:g} TFLOPS x {args.mfu:.0%} MFU")
    print("This is a mathematical estimate, not a trainable model or GPU procurement plan.")


if __name__ == "__main__":
    main()
