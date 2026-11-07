"""GENESIS evaluation harness: one report format for every model and adapter.

    # local base model, then the same base + adapter
    python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B \
        --data data/applied/fin_v1/eval_test.jsonl --name base --gpu-hour-price 2.5 \
        --train-data data/applied/fin_v1/sft_train.jsonl --out reports/applied/base.json
    python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B \
        --adapter runs/applied/fin-v1/adapter --name fin-v1 ... --out reports/applied/fin-v1.json

    # a served model (vLLM / Ollama / any OpenAI-compatible server), the way apps call it
    python -m genesis.harness.evaluate run --endpoint http://localhost:8000/v1 --model fin \
        --data data/applied/fin_v1/eval_test.jsonl --name fin-v1-served --out reports/applied/served.json

    # side-by-side table
    python -m genesis.harness.evaluate compare reports/applied/base.json reports/applied/fin-v1.json

Each item has ``messages`` (the prompt), ``gold`` and optionally ``kind``,
``tolerance`` and ``scorer``. The scorer (``numeric`` by default, see
``genesis.harness.scorers``) decides correctness. Decoding is greedy, so runs
are repeatable. Cost = measured GPU (or endpoint) time x ``--gpu-hour-price``.

Every report carries the same fields: accuracy (overall and per kind), speed,
cost, contamination (exact prompt overlap with ``--train-data``, when given),
the data hash and the GENESIS version. ``report_version`` changes only when
these fields change.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from genesis import __version__ as GENESIS_VERSION
from genesis.adapt.chatdata import load_jsonl, render
from genesis.harness.scorers import is_correct, parse_number, resolve  # noqa: F401  (re-exported)
from genesis.precision import describe, inference_dtype
from genesis.tokenizer import file_sha256

REPORT_VERSION = 1


def summarize(rows: list[dict]) -> dict:
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row["correct"])
    return {
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
        "parsed": sum(r["prediction"] is not None for r in rows) / len(rows) if rows else 0.0,
        "by_kind": {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(by_kind.items())},
    }


def _prompt_key(messages: list[dict]) -> str:
    return "\n".join(" ".join(m["content"].split()) for m in messages if m["role"] == "user")


def contamination(items: list[dict], train_paths: list[Path]) -> dict:
    """Share of eval items whose user prompt appears verbatim (whitespace-normalized) in training data."""
    if not train_paths:
        return {"checked": False}
    seen = {_prompt_key(row["messages"]) for path in train_paths for row in load_jsonl(path)}
    overlap = [item.get("id") for item in items if _prompt_key(item["messages"]) in seen]
    return {"checked": True, "method": "exact user-prompt match",
            "against": [{"path": str(p), "sha256": file_sha256(p)} for p in train_paths],
            "overlap_items": len(overlap), "overlap_rate": len(overlap) / len(items) if items else 0.0,
            "overlap_ids": overlap[:50]}


class LocalGenerator:
    """Hugging Face model (+ optional PEFT adapter) on this machine."""

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.template_kwargs = json.loads(args.template_kwargs)
        self.max_new_tokens = args.max_new_tokens
        self.use_cuda = torch.cuda.is_available() and not args.cpu
        dtype = inference_dtype(torch, self.use_cuda)
        self.precision = describe(torch, self.use_cuda)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        load_kwargs = {"dtype": dtype}
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                    bnb_4bit_compute_dtype=dtype)
        if self.use_cuda:
            load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.hardware = torch.cuda.get_device_name() if self.use_cuda else "cpu"

    def sync(self):
        if self.use_cuda:
            self.torch.cuda.synchronize()

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        tok = self.tokenizer
        prompts = [render(tok, item["messages"], add_generation_prompt=True, template_kwargs=self.template_kwargs)
                   for item in batch]
        encoded = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
        self.sync()
        results = []
        for ids in output[:, encoded["input_ids"].shape[1]:]:
            ids = [t for t in ids.tolist() if t != tok.pad_token_id]
            results.append((tok.decode(ids, skip_special_tokens=True), len(ids)))
        return results


class EndpointGenerator:
    """Any OpenAI-compatible server; the model name is the served base or adapter name."""

    def __init__(self, args):
        from genesis.serving.endpoint import EndpointClient

        self.client = EndpointClient(args.endpoint, api_key=args.api_key)
        self.model = args.model
        self.max_new_tokens = args.max_new_tokens
        self.template_kwargs = json.loads(args.template_kwargs)
        self.hardware = f"endpoint:{args.endpoint}"

    def sync(self):
        pass

    def generate(self, batch: list[dict]) -> list[tuple[str, int]]:
        results = []
        for item in batch:
            reply = self.client.chat(self.model, item["messages"], max_tokens=self.max_new_tokens, temperature=0.0,
                                     template_kwargs=self.template_kwargs or None)
            results.append((reply.text, reply.completion_tokens))
        return results


def run(args) -> dict:
    items = load_jsonl(args.data)
    if args.limit:
        items = items[:args.limit]
    generator = EndpointGenerator(args) if args.endpoint else LocalGenerator(args)
    if args.pipeline == "gorilla":
        from genesis.harness.gorilla import CalcPipeline

        generator = CalcPipeline(generator, digits=args.pipeline_digits, retry=not args.no_pipeline_retry,
                                hint=args.distractor_hint)
    default_scorer = args.scorer
    rows, generated_tokens = [], 0
    generator.sync()
    started = time.perf_counter()
    for begin in range(0, len(items), args.batch_size):
        batch = items[begin:begin + args.batch_size]
        t0 = time.perf_counter()
        outputs = generator.generate(batch)
        batch_seconds = time.perf_counter() - t0
        for item, (text, n_tokens) in zip(batch, outputs):
            generated_tokens += n_tokens
            scorer_name = item.get("scorer", default_scorer)
            prediction, correct = resolve(scorer_name)(text, item)
            rows.append({"id": item.get("id"), "kind": item.get("kind", "unknown"), "lang": item.get("lang"),
                         "scorer": scorer_name, "gold": item["gold"], "prediction": prediction,
                         "correct": bool(correct), "output": text, "seconds": batch_seconds / len(batch)})
    seconds = time.perf_counter() - started
    on_gpu = bool(args.endpoint) or getattr(generator, "use_cuda", False)
    gpu_hours = seconds / 3600 if on_gpu else 0.0
    report = {
        "report_version": REPORT_VERSION, "genesis_version": GENESIS_VERSION,
        "name": args.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "adapter": args.adapter, "endpoint": args.endpoint,
        "data": str(args.data), "data_sha256": file_sha256(args.data),
        "items": len(rows), **summarize(rows),
        "contamination": contamination(items, args.train_data or []),
        "speed": {"total_seconds": round(seconds, 2), "seconds_per_item": round(seconds / max(len(rows), 1), 3),
                  "generated_tokens": generated_tokens,
                  "tokens_per_second": round(generated_tokens / seconds, 1) if seconds else None,
                  "batch_size": args.batch_size},
        "cost": {"gpu_hours": round(gpu_hours, 4), "gpu_hour_price": args.gpu_hour_price,
                 "total": round(gpu_hours * args.gpu_hour_price, 4) if args.gpu_hour_price else None,
                 "per_1000_items": round(gpu_hours * args.gpu_hour_price / max(len(rows), 1) * 1000, 3)
                 if args.gpu_hour_price else None},
        "hardware": generator.hardware,
        "settings": {"max_new_tokens": args.max_new_tokens, "decoding": "greedy", "scorer": default_scorer,
                     "template_kwargs": json.loads(args.template_kwargs), "load_4bit": args.load_4bit,
                     "precision": getattr(generator, "precision", None),
                     "pipeline": args.pipeline, "pipeline_digits": args.pipeline_digits if args.pipeline != "none" else None,
                     "pipeline_fallbacks": getattr(generator, "fallbacks", None),
                     "pipeline_ungrounded": getattr(generator, "ungrounded", None),
                     "pipeline_retried": getattr(generator, "retried", None),
                     "pipeline_rescued": getattr(generator, "rescued", None),
                     "pipeline_hint": getattr(generator, "hint", None)},
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown([report]))
    return report


def to_markdown(reports: list[dict]) -> str:
    kinds = sorted({k for r in reports for k in r["by_kind"]})
    header = ["model", "items", "accuracy", *kinds, "sec/item", "tok/s", "cost/1k", "overlap"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in reports:
        cells = [r["name"], str(r["items"]), f"{r['accuracy']:.1%}"]
        cells += [f"{r['by_kind'][k]['accuracy']:.1%}" if k in r["by_kind"] else "" for k in kinds]
        cost = r["cost"].get("per_1000_items")
        contam = r.get("contamination") or {}
        cells += [f"{r['speed']['seconds_per_item']:.2f}", str(r["speed"]["tokens_per_second"] or ""),
                  f"{cost:.2f}" if cost is not None else "",
                  f"{contam['overlap_rate']:.1%}" if contam.get("checked") else "n/a"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help="HF id / local folder, or the served model name with --endpoint")
    r.add_argument("--adapter", default=None)
    r.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    r.add_argument("--api-key", default=None, help="endpoint key (default: GENESIS_API_KEY env var, if set)")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--train-data", type=Path, nargs="*", default=None,
                   help="training files to check for prompt overlap (contamination)")
    r.add_argument("--scorer", default="numeric", help="default scorer: numeric, exact or module:function")
    r.add_argument("--name", required=True)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--gpu-hour-price", type=float, default=None)
    r.add_argument("--load-4bit", action="store_true")
    r.add_argument("--template-kwargs", default="{}")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--pipeline", choices=["none", "gorilla"], default="none",
                   help="gorilla: the model writes a formula, a calculator computes the answer")
    r.add_argument("--pipeline-digits", type=int, default=2, help="decimals of the calculated answer")
    r.add_argument("--no-pipeline-retry", action="store_true",
                   help="gorilla: do not retry answers worked out without a formula (reproduces the 2026-11-06 numbers)")
    r.add_argument("--distractor-hint", action="store_true",
                   help="gorilla: tell the model to leave numbers that do not belong to the question out of the formula")
    c = sub.add_parser("compare")
    c.add_argument("reports", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.command == "run":
        if args.endpoint and (args.adapter or args.load_4bit):
            p.error("--adapter/--load-4bit apply to local models; with --endpoint pass the served adapter name as --model")
        return run(args)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    print(to_markdown(reports))
    return reports


if __name__ == "__main__":
    main()
