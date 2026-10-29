"""Pretraining loop: single GPU/CPU, or multi-GPU data parallel via torchrun.

    # single device
    python -m genesis.train --config configs/debug.json --data data/processed \
        --out-dir checkpoints/debug --steps 200

    # one node, 8 GPUs (DDP)
    torchrun --standalone --nproc_per_node 8 -m genesis.train \
        --config configs/dense-125m.json --data data/tokens/ko32k-v1 \
        --out-dir checkpoints/dense-125m --batch-size 16 --grad-accum 4 --steps 20000

Features: warmup + cosine LR, AdamW with decay/no-decay groups, gradient
accumulation, BF16 autocast on supporting GPUs, deterministic evaluation over
the validation split reporting loss *and* bits-per-byte, tokens/s and MFU,
atomic checkpoints with exact resume, JSONL metrics.

Batches are drawn from ``np.random.default_rng([seed, rank, step])`` so a
resumed run sees exactly the data it would have seen without interruption.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from .config import ModelConfig
from .data.sft import open_split
from .data.shards import read_manifest
from .estimate import estimate, flops_per_token
from .model import GenesisLM
from .tokenizer import load_tokenizer


# --------------------------------------------------------------------------- utils

def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup to ``max_lr`` then cosine decay to ``max_lr * min_ratio``."""
    if warmup > 0 and step <= warmup:
        return max_lr * step / warmup
    if total <= warmup:
        return max_lr
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay, seen = [], [], set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim >= 2 and "embedding" not in name:
            decay.append(param)
        else:  # norms, biases, (tied) embeddings
            no_decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def batch_rng(seed: int, rank: int, step: int, micro: int) -> np.random.Generator:
    return np.random.default_rng([seed, rank, step, micro])


def atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model: GenesisLM, split, seq_len: int, batch_size: int,
             device: torch.device, byte_lengths: np.ndarray, max_tokens: int | None,
             autocast) -> dict[str, float]:
    """Deterministic pass over (a prefix of) a split: mean loss and bits-per-byte.

    Works for fixed pretraining windows and for variable-length SFT examples
    (targets of -100 are padding or non-assistant tokens and are not scored)."""
    was_training = model.training
    model.eval()
    nll_sum, token_count, byte_count = 0.0, 0, 0
    batch_x, batch_y = [], []

    def flush() -> None:
        nonlocal nll_sum, token_count, byte_count
        length = max(len(x) for x in batch_x)
        x_np = np.zeros((len(batch_x), length), dtype=np.int64)
        y_np = np.full((len(batch_x), length), -100, dtype=np.int64)
        for row, (a, b) in enumerate(zip(batch_x, batch_y)):
            x_np[row, :len(a)] = a
            y_np[row, :len(b)] = b
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        with autocast:
            logits = model(x)["logits"]
        nll = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum", ignore_index=-100)
        valid = y_np != -100
        nll_sum += nll.item()
        token_count += int(valid.sum())
        byte_count += int(byte_lengths[y_np[valid]].sum())
        batch_x.clear()
        batch_y.clear()

    for x, y in split.windows(seq_len, max_tokens):
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) == batch_size:
            flush()
    if batch_x:
        flush()
    if was_training:
        model.train()
    if token_count == 0:
        raise ValueError("evaluation split produced no windows")
    loss = nll_sum / token_count
    return {"loss": loss, "bpb": nll_sum / math.log(2) / max(byte_count, 1),
            "tokens": token_count, "bytes": byte_count}


# --------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/debug.json"))
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/debug"))
    p.add_argument("--steps", type=int, default=50, help="total optimizer steps")
    p.add_argument("--batch-size", type=int, default=4, help="micro-batch per device")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=None, help="default: 2%% of steps (min 1)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-tokens", type=int, default=2_000_000, help="max val tokens per eval")
    p.add_argument("--save-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="continue from out-dir/ckpt_last.pt")
    p.add_argument("--init-from", type=Path, default=None,
                   help="start from these weights (e.g. a pretrained model.pt for SFT); optimizer starts fresh")
    p.add_argument("--stop-after", type=int, default=None,
                   help="exit after this step (schedule still targets --steps); for time-boxed jobs")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dtype", choices=("auto", "bf16", "fp32"), default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="per-device peak dense BF16 TFLOPS for MFU (H100 SXM ~989, A100 ~312)")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_every, args.log_every) < 1:
        raise ValueError("steps, batch size, grad accumulation and intervals must be positive")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() and args.device != "cpu" else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local_rank = 0, 1, 0
    master = rank == 0

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and use_cuda and torch.cuda.is_bf16_supported())
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()

    config = ModelConfig.from_json(args.config)
    manifest = read_manifest(args.data)
    tok_info = manifest.get("tokenizer", {"type": "byte", "spec": "byte", "vocab_size": 257, "eos_id": 256})
    if tok_info["vocab_size"] > config.vocab_size:
        raise ValueError(f"data uses a {tok_info['vocab_size']}-token vocabulary but the model has "
                         f"{config.vocab_size}; set vocab_size in {args.config}")
    tokenizer = load_tokenizer(tok_info["spec"], tok_info.get("sha256"))
    byte_lengths = np.zeros(config.vocab_size, dtype=np.int64)
    byte_lengths[:tokenizer.vocab_size] = tokenizer.token_byte_lengths()
    train_split = open_split(args.data, "train")
    val_split = open_split(args.data, "val")
    stage = manifest.get("stage", "pretrain")
    if stage == "sft" and manifest["max_length"] > config.context_length + 1:
        raise ValueError(f"SFT data allows {manifest['max_length']} tokens per example but the model's "
                         f"context is {config.context_length}; rebuild with --max-length {config.context_length + 1}")

    torch.manual_seed(args.seed)
    model = GenesisLM(config).to(device)
    raw_model = model
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                                  betas=(0.9, args.beta2), eps=1e-8,
                                  **({"fused": True} if use_cuda else {}))
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 50)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    last_path, model_path = args.out_dir / "ckpt_last.pt", args.out_dir / "model.pt"
    metrics_path = args.out_dir / "metrics.jsonl"
    start_step = 0
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state["config"] != asdict(config):
            raise ValueError("checkpoint config differs from --config; refusing to resume")
        raw_model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["torch_rng"].cpu())
        start_step = state["step"]
        if master:
            print(f"resumed from {last_path} at step {start_step}", flush=True)
    else:
        if args.init_from is not None:
            init = torch.load(args.init_from, map_location=device, weights_only=True)
            if init["config"] != asdict(config):
                raise ValueError("--init-from checkpoint has a different model config")
            if (init.get("tokenizer") or {}).get("sha256") != tok_info.get("sha256"):
                raise ValueError("--init-from checkpoint was trained with a different tokenizer")
            raw_model.load_state_dict(init["model_state"])
            if master:
                print(f"initialized weights from {args.init_from} (step {init.get('step')})", flush=True)
        if master and metrics_path.exists():
            metrics_path.unlink()

    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if use_cuda else None)

    seq_len = config.context_length
    tokens_per_step = args.batch_size * args.grad_accum * seq_len * world
    flops_token = flops_per_token(config, seq_len)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if master:
        print(f"device={device} world={world} bf16={use_bf16} params={n_params:,} "
              f"tokenizer={tok_info['type']}({tokenizer.vocab_size}) train_tokens={train_split.total:,} "
              f"tokens/step={tokens_per_step:,} warmup={warmup}", flush=True)
        counts = estimate(config)
        (args.out_dir / "run.json").write_text(json.dumps({
            "config_path": str(args.config), "config": asdict(config), "data": str(args.data),
            "tokenizer": tok_info, "parameters": n_params,
            "non_embedding_active_parameters": counts["non_embedding_active_parameters"],
            "active_parameters": counts["approx_active_parameters_per_token"],
            "stage": stage, "init_from": str(args.init_from) if args.init_from else None,
            "tokens_per_step": tokens_per_step, "steps": args.steps, "world_size": world,
            "train_tokens_available": train_split.total, "lr": args.lr, "seed": args.seed,
            "device": str(device), "bf16": use_bf16,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(record: dict) -> None:
        if master:
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def checkpoint(step: int, val: dict | None) -> None:
        if not master:
            return
        common = {"model_state": raw_model.state_dict(), "config": asdict(config),
                  "tokenizer": tok_info, "step": step, "parameters": n_params,
                  "data": str(args.data), "val": val}
        atomic_save({**common, "optimizer_state": optimizer.state_dict(),
                     "torch_rng": torch.get_rng_state(),
                     "args": {k: str(v) for k, v in vars(args).items()}}, last_path)
        atomic_save(common, model_path)

    model.train()
    last_val = None
    step = start_step
    window_start, window_steps, excluded = time.perf_counter(), 0, 0.0
    window_tokens = 0  # tokens actually processed (SFT batches are shorter than the context)
    last_step = min(args.steps, args.stop_after) if args.stop_after else args.steps
    for step in range(start_step + 1, last_step + 1):
        lr = lr_at(step, args.lr, warmup, args.steps, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        lm_total = balance_total = z_total = 0.0
        loads = None
        for micro in range(args.grad_accum):
            x_np, y_np = train_split.sample(batch_rng(args.seed, rank, step, micro), args.batch_size, seq_len)
            window_tokens += x_np.size * world
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)
            sync = nullcontext()
            if distributed and micro < args.grad_accum - 1:
                sync = model.no_sync()
            with sync:
                with autocast:
                    out = model(x, y)
                (out["loss"] / args.grad_accum).backward()
            lm_total += out["lm_loss"].item() / args.grad_accum
            balance_total += out["balance_loss"].item() / args.grad_accum
            z_total += out["z_loss"].item() / args.grad_accum
            if "expert_load" in out:
                loads = out["expert_load"] if loads is None else loads + out["expert_load"]
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()
        window_steps += 1

        if step % args.log_every == 0 or step == last_step:
            if distributed:  # report the loss averaged over all ranks
                mean_loss = torch.tensor([lm_total], device=device)
                dist.all_reduce(mean_loss)
                lm_total = mean_loss.item() / world
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start - excluded
            tok_s = window_tokens / max(elapsed, 1e-9)
            record = {"step": step, "lr": lr, "train_loss": lm_total, "grad_norm": grad_norm,
                      "tokens_per_s": tok_s, "tokens_seen": step * tokens_per_step}
            if config.is_moe:
                record["balance_loss"] = balance_total
                record["z_loss"] = z_total
                if loads is not None:
                    mean_load = (loads / args.grad_accum).mean(dim=0)
                    record["expert_load_max_over_min"] = float(mean_load.max() / mean_load.min().clamp_min(1e-9))
            if args.peak_tflops:
                record["mfu"] = flops_token * tok_s / world / (args.peak_tflops * 1e12)
            log({"type": "train", **record})
            if master:
                extra = f" mfu={record['mfu']:.1%}" if "mfu" in record else ""
                print(f"step {step}/{args.steps} loss={lm_total:.4f} lr={lr:.2e} "
                      f"gnorm={grad_norm:.2f} tok/s={tok_s:,.0f}{extra}", flush=True)
            window_start, window_steps, excluded = time.perf_counter(), 0, 0.0
            window_tokens = 0

        if step % args.eval_every == 0 or step == args.steps:
            pause = time.perf_counter()
            if master:
                last_val = evaluate(raw_model, val_split, seq_len, args.batch_size, device,
                                    byte_lengths, args.eval_tokens, autocast)
                log({"type": "eval", "step": step, "split": "val", **last_val})
                print(f"step {step}/{args.steps} val_loss={last_val['loss']:.4f} "
                      f"val_bpb={last_val['bpb']:.4f} ({last_val['tokens']:,} tokens)", flush=True)
            if distributed:
                dist.barrier()
            excluded += time.perf_counter() - pause

        if args.save_every and step % args.save_every == 0 and step != args.steps:
            pause = time.perf_counter()
            checkpoint(step, last_val)
            if distributed:
                dist.barrier()
            excluded += time.perf_counter() - pause

    checkpoint(step, last_val)
    if master:
        print(f"saved {model_path} (inference) and {last_path} (resume)", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return {"step": step, "val": last_val, "out_dir": str(args.out_dir)}


if __name__ == "__main__":
    main()
