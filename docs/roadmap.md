# Roadmap — gates to a trainable frontier-scale model

Priority: **Track A (GENESIS-Applied)**. Track B keeps pretraining as a learning track; GENESIS-1B-KO and the 10T experiment stay long-term goals.
Each gate demands test evidence and measurement; no dates or expense guarantees.

## Track A — GENESIS-Applied (priority, career track)
Specialise open models for Korean/Japanese corporate finance with LoRA/QLoRA; publish accuracy, cost and speed.

### A1: data and tasks [CODE DONE — needs DART key]
- [x] OpenDART client (cached, resumable, quota-aware) and statement normalisation
- [x] Extraction and ratio tasks with computed answers; company-level splits; ko/ja templates
- [ ] Collect 300 listed companies × 2023–2025 and review the task manifest

### A2: baseline and first fine-tune [CODE DONE — needs GPU]
- [x] LoRA/QLoRA trainer on any chat template (assistant-only loss), run records
- [x] Evaluator: accuracy by task, seconds/item, tokens/s, cost per 1,000 items
- [ ] Baseline vs fin-v1 on the held-out companies, small model then 8B

### A3: Korean + Japanese, public release [NOT STARTED]
- [ ] EDINET collector and Japanese tasks; KMMLU finance subjects, Japanese Financial Benchmark, KRX-Bench
- [ ] Multi-year trend and peer-comparison tasks
- [ ] Hugging Face release: adapter, model card, evaluation report, data card

## Track B — pretraining from scratch (learning track)

## Gate 0: executable engineering baseline [DONE]
- [x] Self-authored sample data and provenance manifest
- [x] UTF-8 byte-token baseline
- [x] Explicit causal Transformer + top-k SwiGLU MoE
- [x] Single-device pretraining script and checkpoint/inference path
- [x] Correctness tests for causality, gradients and dataset preparation

## Gate 1: trustworthy data + real tokenizer [CODE DONE — needs real corpus]
- [x] Source registry with mandatory license / rights basis / commercial-use fields
- [x] Pluggable source adapters (txt, jsonl(.gz), parquet), NFC normalization
- [x] Quality filters with removal reasons, Korean PII masking, exact + MinHash near-dedup
- [x] Document-level train/val/test splits by content hash, after dedup
- [x] Contamination check against benchmark items + blocklist rebuild
- [x] Byte-level BPE with NFC, per-digit split, reserved chat tokens; tokens-per-syllable report
- [x] Data card and manifests pinned by hash (clean → tokenizer → shards → checkpoint)
- [x] **Assemble a real corpus**: Korean Wikipedia 2026-11-01, 510,989 docs / 0.62B chars after cleaning, data card reviewed
- [x] Train `ko32k-v1`: 0.72 tokens per Hangul syllable vs 3.57 for bytes, 100% round trip
- [ ] Legal/privacy review of the source list

## Gate 2: efficient single-node training [CODE DONE — needs GPU runs]
- [x] RoPE, RMSNorm, GQA, SDPA attention, BF16 autocast
- [x] Optimizer groups, warmup + cosine LR, gradient accumulation, exact resume
- [x] Tokens/sec, MFU, metrics.jsonl; val loss and bits-per-byte
- [x] Single-node multi-GPU via DDP (torchrun)
- [ ] **dense-100m on ~2B tokens**: smooth val bpb, verified resume on GPU, MFU ≥ 30%
- [x] Hugging Face (Llama) export with logit-equality check; lm-evaluation-harness runs on it
- [x] Benchmark item export for KoBEST / HAE-RAE / CLIcK / KMMLU + contamination workflow
- [x] KV cache for generation
- [ ] Measured KoBEST / HAE-RAE / CLIcK scores for dense-100m

## Gate 3: scaling evidence + distributed systems [NOT STARTED]
- [x] Ladder tooling: run plan at fixed tokens-per-parameter, power-law fit, prediction, plot
- [ ] Scaling ladder 40M / 100M / 300M executed; mini scaling law fitted
- [ ] Dense vs MoE at equal active parameters (`dense-100m` vs `moe-8x-100m-active`)
- [ ] FSDP2 sharding; MoE expert parallelism, capacity management, all-to-all transport
- [ ] Checkpoint resharding, fault injection, distributed reproducibility

## Gate 4: serious training runs [NOT STARTED]
- [x] Instruction-tuning path: chat template, masked SFT data, --init-from, chat generation
- [ ] **GENESIS-1B-KO**: dense-1b on 20–50B tokens, then instruction tuning on licensed Korean data
- [ ] Public release: weights, tokenizer, data card, training logs, evaluation + contamination report
- [ ] Raise capital / research partnerships / managed cluster access for 7B → 30B+
- [ ] MoE pilots with reliable training stability, cost and evals; safety evaluation and governance

## Gate 5: GENESIS-10T experiment [ASPIRATIONAL]
- [ ] Architecture selected based on measured efficiency, not just a round number
- [ ] Data volume, scaling-law and compute budget reviewed
- [ ] Cluster interconnect, storage, checkpoint and recovery proved
- [ ] Independent evaluation validates usefulness and safety
- [ ] Report total and active parameter counts and training evidence
