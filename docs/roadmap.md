# Roadmap — gates to a trainable frontier-scale model

Near-term target: **GENESIS-1B-KO** (Gate 4 entry). The 10T experiment stays the north star.
Each gate demands test evidence and measurement; no dates or expense guarantees.

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
- [ ] **Assemble a real corpus** (Korean Wikipedia, statutes, FineWeb2-ko, ...) and review its data card
- [ ] Train `ko32k-v1` on it and record the byte-vs-BPE report
- [ ] Legal/privacy review of the source list

## Gate 2: efficient single-node training [CODE DONE — needs GPU runs]
- [x] RoPE, RMSNorm, GQA, SDPA attention, BF16 autocast
- [x] Optimizer groups, warmup + cosine LR, gradient accumulation, exact resume
- [x] Tokens/sec, MFU, metrics.jsonl; val loss and bits-per-byte
- [x] Single-node multi-GPU via DDP (torchrun)
- [ ] **dense-100m on ~2B tokens**: smooth val bpb, verified resume on GPU, MFU ≥ 30%
- [ ] Benchmark harness: HRET or lm-evaluation-harness on KoBEST / HAE-RAE / CLIcK (needs HF export)
- [ ] KV cache for generation

## Gate 3: scaling evidence + distributed systems [NOT STARTED]
- [ ] Scaling ladder 30M / 100M / 300M at fixed tokens-per-parameter; fit a mini scaling law
- [ ] Dense vs MoE at equal active parameters (`dense-100m` vs `moe-8x-100m-active`)
- [ ] FSDP2 sharding; MoE expert parallelism, capacity management, all-to-all transport
- [ ] Checkpoint resharding, fault injection, distributed reproducibility

## Gate 4: serious training runs [NOT STARTED]
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
