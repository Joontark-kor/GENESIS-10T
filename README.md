# GENESIS-10T

> GENESIS is a three-layer base for Korean-first applied LLM work: an **open base model** (currently Qwen3-4B), the **GENESIS core** (LoRA/QLoRA adapters, one evaluation-report format, an OpenAI-compatible serving contract, a safety-check interface, and the **polarbear** tool layer) and **domain applications** (the first is Korean corporate finance from OpenDART filings).
>
> **Scope since 2026-11-11:** this repository is Track A only. The from-scratch pretraining research (Track B: data pipeline, tokenizer, Transformer/MoE, scaling ladder) moved to the separate repository **ssial**; the two do not import each other, and the earlier history of those files stays here (up to `v0.3.0`).

## What is measured

Polarbear: the model writes a formula (`계산식:`), `genesis.tools` calculates. It is a tool for **structured calculation questions whose values are given** (finance ratios, everyday numbers, structured multi-step). Measured on Qwen3-4B, Colab T4, with ranges over seeds 1-5 and public sets (details, mistakes and decisions: [docs/12_polarbear_ko.md](docs/12_polarbear_ko.md)):

| Set | Baseline | Polarbear |
| --- | --- | --- |
| Finance (544 items, one run) | 85.3% | 98.7% |
| Multi-step (240, seeds 1-5) | 78.7% | 88.8% |
| Everyday numbers (420, seeds 1-5) | 93.1% | 97.0% |
| No-calculation questions (240, seeds 1-5) | 85.1% | 79.2% (lower; call it only on calculation questions) |
| Korean GSM8K, full (1319, one run) | 64.1% | 59.7% (lower) |

Fine-tuning (QLoRA) did not help on the finance set (84.0%): the errors were the model's own arithmetic.

## Quick start (CPU is enough)

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

GPU work (Colab T4 is enough for the 4B model): `pip install -e ".[fin,qlora]"`, then the notebooks and `scripts/colab_*.sh`.

## Guides

[Core contract](docs/11_core_ko.md) · [finance application and terminal demo](docs/10_fin_ko.md) · [polarbear measurements](docs/12_polarbear_ko.md) · [prediction engine (research, off by default)](docs/13_predict_ko.md) · [roadmap](docs/roadmap.md)

## Components

| Path | Purpose |
| --- | --- |
| `genesis/adapt/` | LoRA/QLoRA fine-tuning of an open base model on chat data (assistant-only loss), `run.json` provenance |
| `genesis/harness/` | One evaluation-report format for every model/adapter: accuracy, speed, cost, contamination; local or served models; the polarbear pipeline and scorers |
| `genesis/serving/` | OpenAI-compatible endpoint client, vLLM multi-LoRA command, `CalcClient`, a test-only transformers server |
| `genesis/tools.py` | The calculator the model's formulas go through |
| `genesis/safety.py` | Safety-check interface only; implementations live with each application |
| `genesis/hubsync.py`, `precision.py`, `util.py` | Private Hugging Face sync for Colab, bf16/fp16 choice, small shared helpers |
| `genesis_fin/` | Domain package: OpenDART client, financial tasks with computed answers, the terminal demo. Depends on `genesis`; never the reverse |
| `genesis_calc/` | Everyday-number task generators and the external benchmark loaders (evaluation only) |
| `genesis_predict/` | Prediction-engine research track (off by default) |
| `notebooks/`, `scripts/` | Colab runbooks and run scripts |

## Data governance

Use only datasets you own or whose terms permit the intended use. External benchmarks (KMMLU, Korean GSM8K) are downloaded at run time for evaluation only and are never committed or redistributed.

License for **code**: MIT. Dataset licensing must be evaluated separately.
