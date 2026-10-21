"""Parameter-count estimates without allocating a huge model.

Counts correspond to the simple reference architecture in genesis.model;
not to an industry frontier model with other architectural components.
"""
from __future__ import annotations

import argparse
from .config import ModelConfig


def estimate(config: ModelConfig) -> dict[str, int]:
    d, l, e, h = config.d_model, config.n_layers, config.n_experts, config.expert_hidden_dim
    emb = (config.vocab_size + config.context_length) * d  # tied LM output weights
    attn_per_layer = 4 * d * d + 4 * d
    norms_per_layer = 4 * d
    router_per_layer = d * e
    experts_per_layer = 3 * d * h * e
    final_norm = 2 * d
    total = emb + l * (attn_per_layer + norms_per_layer + router_per_layer + experts_per_layer) + final_norm
    # Active expert estimate assumes k of E experts per token, while router,
    # attention, position embeddings, output head remain fully evaluated.
    active = total - l * (e - config.experts_per_token) * (3 * d * h)
    return {"total_parameters": total, "approx_active_parameters_per_token": active,
            "bf16_weights_bytes": 2 * total, "adam_example_state_bytes": 16 * total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/debug.json")
    parser.add_argument("--hypothetical-10t", action="store_true")
    args = parser.parse_args()
    if args.hypothetical_10t:
        config = ModelConfig(vocab_size=131072, context_length=8192, n_layers=96,
                             n_heads=64, d_model=8192, n_experts=256,
                             experts_per_token=2, expert_hidden_dim=16384)
    else:
        config = ModelConfig.from_json(args.config)
    for key, value in estimate(config).items():
        print(f"{key}: {value:,}")
    print("This is a mathematical size estimate, not a trainable model or GPU procurement plan.")


if __name__ == "__main__":
    main()
