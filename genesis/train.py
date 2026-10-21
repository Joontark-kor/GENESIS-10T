"""Single-node learning baseline. Not capable of training 10T parameters."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from .config import ModelConfig
from .model import GenesisLM


def sample_batch(tokens: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= seq_len:
        raise ValueError(f"need > {seq_len} tokens in each split, got {len(tokens)}")
    starts = np.random.randint(0, len(tokens) - seq_len, size=batch_size)
    x = np.stack([tokens[i:i+seq_len] for i in starts]).astype(np.int64)
    y = np.stack([tokens[i+1:i+seq_len+1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model: GenesisLM, tokens: np.ndarray, seq_len: int, batch_size: int,
             device: torch.device, iterations: int = 3) -> float:
    model.eval()
    losses = []
    for _ in range(iterations):
        x, y = sample_batch(tokens, seq_len, batch_size, device)
        losses.append(model(x, y)["lm_loss"].item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/debug.json"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/debug.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("steps, batch size and eval interval must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    config = ModelConfig.from_json(args.config)
    train_tokens = np.memmap(args.data / "train.bin", dtype=np.uint16, mode="r")
    val_tokens = np.memmap(args.data / "val.bin", dtype=np.uint16, mode="r")
    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"device={device}, parameters={parameter_count:,} (debug baseline, NOT 10T)", flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(train_tokens, config.context_length, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(x, y)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_tokens, config.context_length, args.batch_size, device)
            print(f"step {step}/{args.steps} train_lm_loss={output['lm_loss'].item():.4f} "
                  f"val_lm_loss={val_loss:.4f} balance={output['balance_loss'].item():.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save state only. Inference loads with weights_only=True.
    torch.save({"model_state": model.state_dict(), "config": asdict(config),
                "steps": args.steps, "parameters": parameter_count}, args.output)
    print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
