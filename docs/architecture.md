# Architecture decision record — GENESIS-10T

## What exists now (verified milestone)

A causal decoder-only Transformer with rotary-free learned positions, explicit masked self-attention, pre-normalization, sparse top-k MoE SwiGLU experts, an auxiliary router-balancing term and tied token/input embeddings. The reference implementation is correct-oriented rather than optimized.

## Long-term hypothesis (NOT implemented)

Target: approximately 10 trillion **total** parameters with far fewer **active** parameters per token.

One approximate sizing thought experiment:

| Component | Hypothetical size |
| --- | ---: |
| Layers | 96 |
| Model width | 8,192 |
| Attention heads | 64 |
| Experts per layer | 256 |
| Active experts per token | 2 |
| SwiGLU expert hidden width | 16,384 |
| Expert weights (3 * width * hidden * experts * layers) | ~9.896T |
| Attention projections (4 * width² * layers) | ~25.8B |
| Approximate total excluding norms/embeddings | ~9.92T |
| Approximate activated expert+attention weights | ~103B |

This is a **back-of-envelope hypothesis**, not a production architecture specification. It omits embeddings, sequence routing, capacity/communication overhead, router details, attention cache, normalization and other parameters. It does not demonstrate that training at this scale is feasible or optimal. Expert partitioning plus tensor and pipeline parallelism, low-precision kernels, robust data pipelines and training stability must be engineered and benchmarked.

## Memory sanity check

10T parameters x 2 bytes/BF16 = 20TB *weight storage* only. A simplified mixed-precision Adam-style training-state estimate can be ~16 bytes/parameter = 160TB before activations, communication buffers, temporary kernels, checkpoint copies or redundancy. Memory-only divisions by GPU HBM capacity do **not** constitute a workable GPU plan; throughput, interconnect and reliability constraints are decisive.

## Recommended future technology evaluation

- PyTorch distributed basics → DDP/FSDP2 for dense shards, not expert sharding by itself.
- Megatron Core / DeepSpeed MoE style expert parallelism and all-to-all routing.
- FlashAttention / fused kernels / FP8/BF16 mixed precision where validated.
- Fault-tolerant distributed checkpoints; restart and data-loader determinism.
- Benchmark multiple routes: MoE versus dense, compute-optimal vs parameter-maximal.

## Scientific milestones

No claim of 'largest' without an auditable definition (total vs active parameters), a credible training run, externally reproducible evidence, and a specified comparison date and model set. A model only described in a configuration file has not been trained.
