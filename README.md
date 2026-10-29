# GENESIS-10T

> Research ambition: a 10-trillion-parameter sparse Mixture-of-Experts foundation model.
>
> **Near-term target: GENESIS-1B-KO** — a ~1B-parameter Korean-first model trained from scratch on data whose license is tracked per document, released with its data card, tokenizer and reproduction scripts.
>
> **Actually implemented:** the full small-scale pipeline (data registry → cleaning → Korean BPE → token shards → dense/MoE training with DDP → evaluation). No large model has been trained yet.

## Objective

Build the engineering capability to train a language model from **legally usable raw data**, with provenance, a tokenizer, a causally masked Transformer, sparse top-k expert routing, pretraining, evaluation and eventual scale-out.

10T describes **total** parameters, not activated parameters. Scaling requires measured quality, safety, training throughput, distributed expertise, a team, significant financing and GPU infrastructure. Larger size alone is not a useful quality metric.

## Quick start (CPU is enough)

Requires Python 3.10+ and PyTorch. Prefer Linux/WSL2 for GPU/distributed work.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q

python -m genesis.prepare --input data/raw --output data/processed \
    --source self-authored-demo --license self-authored
python -m genesis.train --config configs/debug.json --data data/processed \
    --out-dir checkpoints/debug --steps 25 --eval-every 5
python -m genesis.generate --checkpoint checkpoints/debug/model.pt --prompt "인공지능은" --tokens 60
```

The demo corpus is tiny, so generation is nonsense; the point is that every stage runs and is tested.

## The real pipeline

See **[docs/02_pipeline_ko.md](docs/02_pipeline_ko.md)** for the step-by-step guide and **[docs/03_next_steps_ko.md](docs/03_next_steps_ko.md)** for what to do next.

```bash
# A. sources -> normalized, filtered, PII-masked, deduplicated, split text (+ DATA_CARD.md)
python -m genesis.data.clean --sources configs/sources.json --output data/clean/v1
# B. Korean-first byte-level BPE, trained on the train split only
python -m genesis.train_tokenizer --input data/clean/v1 --vocab-size 32768 --output tokenizers/ko32k-v1.json
# C. token shards pinned to the tokenizer hash
python -m genesis.data.tokenize --input data/clean/v1 --tokenizer tokenizers/ko32k-v1.json --output data/tokens/ko32k-v1
# D. train (single GPU, or torchrun for multi-GPU)
python -m genesis.train --config configs/dense-100m.json --data data/tokens/ko32k-v1 \
    --out-dir checkpoints/dense-100m --batch-size 16 --grad-accum 4 --steps 15000 --save-every 1000
# E. evaluate and check benchmark contamination
python -m genesis.evaluate --checkpoint checkpoints/dense-100m/model.pt --data data/tokens/ko32k-v1 --split val
python -m genesis.contamination --train data/clean/v1 --benchmark bench.jsonl --fields question
```

## Components

| Path | Purpose |
| --- | --- |
| `genesis/text.py` | NFC normalization (fixes decomposed Hangul), script ratios |
| `genesis/data/sources.py` | Source registry; refuses sources without license and rights basis |
| `genesis/data/filters.py`, `pii.py`, `dedup.py` | Quality heuristics with reasons, Korean PII masking, exact + MinHash dedup |
| `genesis/data/clean.py` | Stage A: clean text, content-hash splits, `DATA_CARD.md` |
| `genesis/tokenizer.py`, `train_tokenizer.py`, `tokenizer_report.py` | Byte and BPE tokenizers, training, tokens-per-syllable report |
| `genesis/data/tokenize.py`, `shards.py` | Stage C: sharded uint16/uint32 token files and reader |
| `genesis/model.py` | RMSNorm, RoPE, GQA attention (SDPA), dense SwiGLU or top-k MoE |
| `genesis/train.py` | LR schedule, grad accumulation, BF16, DDP, exact resume, bpb, MFU |
| `genesis/evaluate.py`, `contamination.py` | Held-out loss/bpb; benchmark n-gram contamination + blocklist |
| `genesis/estimate.py` | Exact parameter counts, training FLOPs and GPU-hours |
| `configs/` | `debug`, `ko-tiny-cpu`, `dense-100m`, `moe-8x-100m-active`, `dense-300m`, `dense-1b`, `sources.example.json` |

## Data governance

Use only datasets you own or whose terms permit the intended training and redistribution. The registry makes you record license, rights basis and commercial-use status for every source, and the data card reports what was kept and removed. **The pipeline cannot verify that what you record is true.** Regex PII masking misses names, addresses and account numbers. Get legal review before any commercial release.

## Important boundaries

- No FSDP/tensor/pipeline/expert parallelism yet: DDP only, so models must fit on one GPU (≈ up to 1B on 80 GB).
- MoE routing is a single-device loop, correct but not fast.
- No KV cache in generation, no instruction tuning, no downstream benchmark harness integration yet.
- Never infer quality from parameter count alone; verify against held-out benchmarks.

License for **code**: MIT. Dataset licensing must be evaluated separately.

## 10T parameter arithmetic without allocating a model

```bash
python -m genesis.estimate --config configs/dense-1b.json --tokens 3e10   # size + GPU-hours
python -m genesis.estimate --hypothetical-10t
```
