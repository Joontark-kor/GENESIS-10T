"""Sample text from a trained checkpoint with the tokenizer it was trained on."""
import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GenesisLM
from .tokenizer import load_tokenizer


def load_model(checkpoint_path: Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = GenesisLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    info = checkpoint.get("tokenizer") or {"spec": "byte"}
    tokenizer = load_tokenizer(info["spec"], info.get("sha256"))
    return model.eval(), tokenizer, checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/debug/model.pt"))
    p.add_argument("--prompt", default="인공지능은")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chat", action="store_true", help="wrap the prompt as a user turn (SFT models)")
    p.add_argument("--system", default=None, help="optional system message with --chat")
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer, checkpoint = load_model(args.checkpoint, args.device)
    stop_id = tokenizer.eos_id
    if args.chat:
        from .chat import IM_END, generation_prompt
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": args.prompt})
        prompt_ids = generation_prompt(tokenizer, messages)
        stop_id = tokenizer.token_id(IM_END)
    else:
        prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt must contain text")
    top_k = min(args.top_k, model.config.vocab_size) if args.top_k else None
    generated = model.generate(torch.tensor([prompt_ids], device=args.device), new_tokens=args.tokens,
                               temperature=args.temperature, top_k=top_k, eos_id=stop_id)
    ids = generated[0].tolist()
    print(tokenizer.decode(ids[len(prompt_ids):]) if args.chat else tokenizer.decode(ids))
    if checkpoint.get("parameters", 0) < 50_000_000:
        print("\n[Small model: output is not expected to be coherent Korean.]")


if __name__ == "__main__":
    main()
