"""Generate tokens from a locally trained checkpoint."""
import argparse
from pathlib import Path
import torch

from .config import ModelConfig
from .model import GenesisLM
from .tokenizer import ByteTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/debug.pt"))
    p.add_argument("--prompt", default="인공지능은")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["config"])
    model = GenesisLM(config)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain text")
    generated = model.generate(torch.tensor([prompt_ids]), new_tokens=args.tokens,
                               temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(generated[0].tolist()))
    print("\n[Debug model: output is not expected to be coherent Korean.]")


if __name__ == "__main__":
    main()
