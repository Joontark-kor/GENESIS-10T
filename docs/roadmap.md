# Roadmap — gates to a trainable frontier-scale model

## Gate 0: executable engineering baseline [IMPLEMENTED]
- [x] Self-authored sample data and provenance manifest
- [x] UTF-8 byte-token baseline
- [x] Explicit causal Transformer + top-k SwiGLU MoE
- [x] Single-device pretraining script and checkpoint/inference path
- [x] Correctness tests for causality, gradients and dataset preparation

## Gate 1: trustworthy data + real tokenizer [NOT STARTED]
- [ ] Data permissions, licensing registry, privacy review
- [ ] Pluggable source adapters with robust decoding, near-duplicate detection
- [ ] Document-level train/val/test splits and contamination checks
- [ ] Train BPE/unigram tokenizer with multilingual coverage metrics
- [ ] Reproducible dataset snapshots and token statistics

## Gate 2: efficient single-node training [NOT STARTED]
- [ ] RoPE/long-context policy, BF16 AMP, scaled dot product attention
- [ ] Optimizer groups, LR schedule, gradient accumulation, resume
- [ ] Tokens/sec, utilization, gradient/activation memory telemetry
- [ ] Standard language modeling and benchmark evaluations

## Gate 3: distributed systems [NOT STARTED]
- [ ] Multi-GPU dense baseline; verified numerical equivalence tests
- [ ] MoE expert parallelism, capacity management, all-to-all transport
- [ ] Tensor and pipeline parallelism, checkpoint resharding and fault injection
- [ ] Security, governance and distributed run reproducibility

## Gate 4: serious training runs [NOT STARTED]
- [ ] 1B → 7B → 30B+ parameter trials with measured budget and quality gates
- [ ] MoE pilots with reliable training stability, cost and evals
- [ ] Raise capital / research partnerships / managed cluster access
- [ ] Pretrain, post-train, safety evaluation and governance

## Gate 5: GENESIS-10T experiment [ASPIRATIONAL]
- [ ] Architecture selected based on measured efficiency, not just a round number
- [ ] Data volume, scaling-law and compute budget reviewed
- [ ] Cluster interconnect, storage, checkpoint and recovery proved
- [ ] Independent evaluation validates usefulness and safety
- [ ] Report total and active parameter counts and training evidence

Each gate demands test evidence and measurement; no dates or expense guarantees.
