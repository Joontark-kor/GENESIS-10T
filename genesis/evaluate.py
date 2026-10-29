"""Evaluate a checkpoint's language-modeling quality on a held-out split.

    python -m genesis.evaluate --checkpoint checkpoints/dense-100m/model.pt \
        --data data/tokens/ko32k-v1 --split test

Reports mean token loss, perplexity and bits-per-byte. Compare models with
*different tokenizers* using bits-per-byte only; token loss is not comparable.
Use the test split sparingly (final numbers), and the val split while tuning.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data.shards import TokenSplit, read_manifest
from .generate import load_model
from .train import evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", type=Path, help="write the result here")
    args = p.parse_args()

    model, tokenizer, checkpoint = load_model(args.checkpoint, args.device)
    data_tok = read_manifest(args.data).get("tokenizer", {})
    if data_tok.get("sha256") != checkpoint.get("tokenizer", {}).get("sha256") or \
            data_tok.get("type") != checkpoint.get("tokenizer", {}).get("type"):
        raise ValueError("checkpoint and dataset were produced with different tokenizers")
    byte_lengths = np.zeros(model.config.vocab_size, dtype=np.int64)
    byte_lengths[:tokenizer.vocab_size] = tokenizer.token_byte_lengths()
    use_bf16 = args.device.startswith("cuda") and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    result = evaluate(model, TokenSplit(args.data, args.split), model.config.context_length,
                      args.batch_size, torch.device(args.device), byte_lengths, args.max_tokens, autocast)
    result.update({"perplexity": math.exp(result["loss"]), "split": args.split,
                   "checkpoint": str(args.checkpoint), "step": checkpoint.get("step"),
                   "parameters": checkpoint.get("parameters")})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
