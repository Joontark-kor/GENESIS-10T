"""LoRA / QLoRA fine-tuning of an open model on chat data (assistant-only loss).

    python -m genesis.adapt.finetune --base-model Qwen/Qwen3-8B \
        --train data/applied/fin_v1/sft_train.jsonl --val data/applied/fin_v1/sft_val.jsonl \
        --output runs/applied/qwen3-8b-fin-v1 --qlora --epochs 2

``--qlora`` loads the base model in 4-bit (bitsandbytes, NVIDIA GPU) so an
8B model trains on one 24 GB card; without it LoRA trains on a bf16/fp16/fp32 copy.
Precision follows the GPU: bf16 on Ampere or newer (A100, L4), fp16 on
older cards such as the free-Colab T4, fp32 on CPU.
The output folder holds the adapter, the tokenizer and ``run.json``
(base model and revision, data hashes, settings, time, peak memory, trainable
parameters). Check the base model's license on its model card before
publishing anything trained from it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from ..precision import describe, native_bf16
from ..util import file_sha256
from .chatdata import Collator, encode_example, load_jsonl


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="Hugging Face id or local folder")
    p.add_argument("--revision", default=None, help="pin a model revision (commit) for reproducibility")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs when > 0")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear",
                   help='"all-linear" or comma-separated names such as q_proj,v_proj')
    p.add_argument("--qlora", action="store_true", help="4-bit base weights (needs bitsandbytes + CUDA)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --output")
    p.add_argument("--merge", action="store_true", help="also save base+adapter merged weights")
    p.add_argument("--template-kwargs", default="{}",
                   help='extra chat-template arguments as JSON, e.g. {"enable_thinking": false}')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU (tests and smoke runs)")
    return p


def encode_file(tokenizer, path: Path, max_length: int, template_kwargs: dict) -> tuple[list[dict], dict]:
    rows = load_jsonl(path)
    examples, skipped = [], 0
    for row in rows:
        item = encode_example(tokenizer, row["messages"], max_length, template_kwargs)
        if item is None:
            skipped += 1
        else:
            examples.append(item)
    return examples, {"rows": len(rows), "used": len(examples), "skipped": skipped,
                      "loss_tokens": sum(sum(1 for l in e["labels"] if l != -100) for e in examples),
                      "sha256": file_sha256(path)}


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    template_kwargs = json.loads(args.template_kwargs)
    use_cuda = torch.cuda.is_available() and not args.cpu
    use_bf16 = native_bf16(torch, use_cuda)
    use_fp16 = use_cuda and not use_bf16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    if tokenizer.chat_template is None:
        raise ValueError("the base model has no chat template; pick an instruct/chat model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples, train_stats = encode_file(tokenizer, args.train, args.max_length, template_kwargs)
    val_examples, val_stats = (encode_file(tokenizer, args.val, args.max_length, template_kwargs)
                               if args.val else ([], None))
    if not train_examples:
        raise ValueError(f"no usable training examples in {args.train} (all too long or unmatched)")
    print(f"train: {train_stats['used']}/{train_stats['rows']} examples, {train_stats['loss_tokens']:,} loss tokens"
          + (f"; val: {val_stats['used']}/{val_stats['rows']}" if val_stats else ""), flush=True)

    load_kwargs: dict = {"revision": args.revision}
    if args.qlora:
        if not use_cuda:
            raise ValueError("--qlora needs an NVIDIA GPU; drop --qlora to train LoRA on CPU")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16)
    # fp16 runs keep fp32 master weights (LoRA params must not be fp16 under the fp16 grad scaler)
    load_kwargs["dtype"] = torch.bfloat16 if use_bf16 else torch.float32
    if use_cuda:
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    targets = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable / total:.2%})", flush=True)

    steps_per_epoch = math.ceil(len(train_examples) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    training_args = TrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        logging_steps=args.logging_steps, save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps" if val_examples else "no", eval_steps=args.eval_steps,
        bf16=use_bf16, fp16=use_fp16, use_cpu=not use_cuda, report_to="none", remove_unused_columns=False,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch", seed=args.seed,
        gradient_checkpointing=False, label_names=["labels"])
    trainer = Trainer(model=model, args=training_args, train_dataset=train_examples,
                      eval_dataset=val_examples or None, data_collator=Collator(tokenizer.pad_token_id),
                      processing_class=tokenizer)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    seconds = time.perf_counter() - started
    final_eval = trainer.evaluate() if val_examples else {}

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output / "merged")
        tokenizer.save_pretrained(args.output / "merged")
    base_revision = getattr(model.base_model.model.config, "_commit_hash", None) or args.revision
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "adapt-finetune",
        "base_model": args.base_model, "base_revision": base_revision,
        "method": "QLoRA (4-bit nf4)" if args.qlora else "LoRA", "precision": describe(torch, use_cuda),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "data": {"train": train_stats, "val": val_stats},
        "trainable_parameters": trainable, "total_parameters": total,
        "train_loss": result.training_loss, "eval": final_eval, "steps": result.global_step,
        "train_seconds": round(seconds, 1),
        "gpu": torch.cuda.get_device_name() if use_cuda else None,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if use_cuda else None,
        "license_note": "Check the base model's license before publishing the adapter or merged weights.",
    }
    (args.output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("train_loss", "eval", "steps", "train_seconds", "peak_gpu_memory_gb")},
                     indent=2), flush=True)
    return run


if __name__ == "__main__":
    main()
