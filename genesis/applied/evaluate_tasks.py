"""Accuracy, speed and cost of a base or fine-tuned model on financial tasks.

    # base model
    python -m genesis.applied.evaluate_tasks run --model Qwen/Qwen3-8B \
        --data data/applied/fin_v1/eval_test.jsonl --name base --gpu-hour-price 2.5 \
        --out reports/applied/base.json
    # same base + adapter
    python -m genesis.applied.evaluate_tasks run --model Qwen/Qwen3-8B \
        --adapter runs/applied/qwen3-8b-fin-v1/adapter --name fin-v1 ... --out reports/applied/fin-v1.json
    # side-by-side table
    python -m genesis.applied.evaluate_tasks compare reports/applied/base.json reports/applied/fin-v1.json

Scoring reads the number after "정답:" / "答え:" / "Answer:" (else the last
number in the reply) and accepts it within the item's tolerance. Decoding is
greedy, so runs are repeatable. Cost = measured GPU time x ``--gpu-hour-price``
(your provider's hourly price; currency as you enter it).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from ..tokenizer import file_sha256
from .chatdata import load_jsonl, render

_TAGGED = re.compile(r"(?:정답|答え|Answer)\s*[:：]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"[-−△]?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """Number after an answer tag, else the last number in the text."""
    tagged = _TAGGED.findall(text)
    source = tagged[-1] if tagged else text
    numbers = _NUMBER.findall(source)
    if not numbers:
        return None
    raw = numbers[0] if tagged else numbers[-1]
    negative = raw[0] in "-−△"
    try:
        value = float(raw.lstrip("-−△").replace(",", ""))
    except ValueError:
        return None
    return -value if negative else value


def is_correct(prediction: float | None, gold: float, tolerance: float) -> bool:
    return prediction is not None and abs(prediction - gold) <= tolerance + 1e-9


def summarize(rows: list[dict]) -> dict:
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row["correct"])
    return {
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "parsed": sum(r["prediction"] is not None for r in rows) / len(rows) if rows else 0.0,
        "by_kind": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by_kind.items())},
    }


def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_jsonl(args.data)
    if args.limit:
        items = items[:args.limit]
    template_kwargs = json.loads(args.template_kwargs)
    use_cuda = torch.cuda.is_available() and not args.cpu
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kwargs = {"dtype": dtype}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                bnb_4bit_compute_dtype=dtype)
    if use_cuda:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    rows, generated_tokens = [], 0
    if use_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    for begin in range(0, len(items), args.batch_size):
        batch = items[begin:begin + args.batch_size]
        prompts = [render(tokenizer, item["messages"], add_generation_prompt=True, template_kwargs=template_kwargs)
                   for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        if use_cuda:
            torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - t0
        new = output[:, encoded["input_ids"].shape[1]:]
        for item, ids in zip(batch, new):
            ids = [t for t in ids.tolist() if t != tokenizer.pad_token_id]
            generated_tokens += len(ids)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            prediction = parse_number(text)
            rows.append({"id": item.get("id"), "kind": item.get("kind", "unknown"), "lang": item.get("lang"),
                         "gold": item["gold"], "prediction": prediction,
                         "correct": is_correct(prediction, float(item["gold"]), float(item.get("tolerance", 0))),
                         "output": text, "seconds": batch_seconds / len(batch)})
    seconds = time.perf_counter() - started
    summary = summarize(rows)
    gpu_hours = seconds / 3600 if use_cuda else 0.0
    report = {
        "name": args.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "adapter": args.adapter, "data": str(args.data), "data_sha256": file_sha256(args.data),
        "items": len(rows), **summary,
        "speed": {"total_seconds": round(seconds, 2), "seconds_per_item": round(seconds / max(len(rows), 1), 3),
                  "generated_tokens": generated_tokens,
                  "tokens_per_second": round(generated_tokens / seconds, 1) if seconds else None,
                  "batch_size": args.batch_size},
        "cost": {"gpu_hours": round(gpu_hours, 4), "gpu_hour_price": args.gpu_hour_price,
                 "total": round(gpu_hours * args.gpu_hour_price, 4) if args.gpu_hour_price else None,
                 "per_1000_items": round(gpu_hours * args.gpu_hour_price / max(len(rows), 1) * 1000, 3)
                 if args.gpu_hour_price else None},
        "hardware": torch.cuda.get_device_name() if use_cuda else "cpu",
        "settings": {"max_new_tokens": args.max_new_tokens, "decoding": "greedy",
                     "template_kwargs": template_kwargs, "load_4bit": args.load_4bit},
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown([report]))
    return report


def to_markdown(reports: list[dict]) -> str:
    kinds = sorted({k for r in reports for k in r["by_kind"]})
    header = ["model", "items", "accuracy", *kinds, "sec/item", "tok/s", "cost/1k"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in reports:
        cells = [r["name"], str(r["items"]), f"{r['accuracy']:.1%}"]
        cells += [f"{r['by_kind'][k]['accuracy']:.1%}" if k in r["by_kind"] else "" for k in kinds]
        cost = r["cost"].get("per_1000_items")
        cells += [f"{r['speed']['seconds_per_item']:.2f}", str(r["speed"]["tokens_per_second"] or ""),
                  f"{cost:.2f}" if cost is not None else ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--adapter", default=None)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--gpu-hour-price", type=float, default=None)
    r.add_argument("--load-4bit", action="store_true")
    r.add_argument("--template-kwargs", default="{}")
    r.add_argument("--cpu", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("reports", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.command == "run":
        return run(args)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    print(to_markdown(reports))
    return reports


if __name__ == "__main__":
    main()
