# GENESIS-10T

> Research ambition: a 10-trillion-parameter sparse Mixture-of-Experts foundation model.
>
> **Actual implemented milestone:** a small, single-device, causal MoE Transformer for learning and validation. It is **not** a 10T model and is not yet distributed or optimized for scale.

## Objective

Build the engineering capability to train a language model from **legally usable raw data**, with provenance, a tokenizer, a causally masked Transformer, sparse top-k expert routing, pretraining, evaluation and eventual scale-out.

10T describes **total** parameters, not necessarily activated parameters. Scaling requires empirical quality, safety, training throughput, distributed expertise, a team, significant financing and appropriate GPU infrastructure. Larger size alone is not a useful quality metric.

## Quick start (Windows / PowerShell or Linux)

Requires Python 3.10+ and PyTorch. Prefer Linux/WSL2 for future GPU/distributed work.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest -q
python -m genesis.prepare --input data/raw --output data/processed --source self-authored-demo --license self-authored
python -m genesis.train --steps 25 --batch-size 4 --eval-every 5
python -m genesis.generate --prompt "인공지능은" --tokens 60
```

On Linux/WSL2 use `source .venv/bin/activate` instead of the PowerShell activation line. If PowerShell execution policy prevents activation, use `.venv\Scripts\python.exe -m ...` or Command Prompt activation.

For CUDA hardware, follow the device-specific installer at https://pytorch.org/get-started/locally/ . CPU mode works for small tests.

### Expected behavior

- `pytest`: all tests pass, including strict causal masking and gradient flow.
- `prepare`: outputs `data/processed/{train.bin,val.bin,manifest.json}`.
- `train`: prints parameter count, language modeling loss and saves `checkpoints/debug.pt`.
- `generate`: emits text, most likely **nonsense**. The supplied dataset is intentionally tiny, and byte-token language generation needs much more data.

## Components

| Path | Purpose |
| --- | --- |
| `genesis/prepare.py` | Text cleaning, dedup, split and provenance metadata |
| `genesis/tokenizer.py` | Reversible UTF-8 byte tokenizer baseline |
| `genesis/model.py` | Causal attention, SwiGLU experts, sparse MoE routing, LM |
| `genesis/train.py` | Single-device AdamW training, validation and checkpoint |
| `genesis/generate.py` | Load checkpoint and sample tokens |
| `configs/debug.json` | Debug model configuration |
| `tests/` | Causality, gradients, tokenizer, data pipeline checks |
| `docs/architecture.md` | Model design and scaling boundaries |
| `docs/roadmap.md` | Ordered, measurable project milestones |

## Data governance

Use only datasets you own or whose terms explicitly permit the intended training and redistribution. Keep original license/provenance records. **The starter data pipeline does not automatically remove personal information, copyrighted material, or harmful content**, and does not prove license validity. Do not crawl arbitrary sites and presume permission.

## Important boundaries

- No tokenizer training yet: byte tokenizer is a deterministic bootstrap, not a production BPE.
- No DDP/FSDP, tensor/pipeline/expert parallelism, all-to-all routing or GPU kernels yet.
- No resilient multi-node checkpointing, throughput benchmarking, safety tests or dataset scale validation yet.
- Never infer quality from parameter count alone; verify against held-out benchmarks.

License for **code**: MIT. Dataset licensing must be evaluated separately.

## 10T parameter arithmetic without allocating a model

```bash
python -m genesis.estimate                         # debug model count
python -m genesis.estimate --hypothetical-10t      # hypothetical 10T architecture
```

This computes theoretical counts only; it does not reserve GPU memory or train the model.
