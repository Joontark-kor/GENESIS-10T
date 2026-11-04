# AGENTS.md — GENESIS-10T

Guide for coding agents (Codex, Claude Code, etc.). Read this first, then `docs/11_core_ko.md` (core contract) and `docs/10_fin_ko.md` (first application).

## What this repo is

GENESIS is a three-layer base for the owner's work: (1) base weights (open model, currently Qwen3-4B; research track: own pretraining), (2) **GENESIS core**, (3) applications (domain LoRA adapters). The standard is a **convention**, not weights:

- apps call only an **OpenAI-compatible endpoint**; they never import training code,
- one **evaluation-report format** (`REPORT_VERSION = 1`) for every adapter,
- versions are **frozen with tags**; applications pin a version,
- the **safety check is an interface only** (`genesis/safety.py`); implementations live outside this repo.

## Layout

| Path | Role |
|---|---|
| `genesis/adapt/` | data format + LoRA/QLoRA fine-tuning (`python -m genesis.adapt.finetune`) |
| `genesis/harness/` | evaluation harness + scorers (`python -m genesis.harness.evaluate run\|compare`) |
| `genesis/serving/` | OpenAI-compatible endpoint (vLLM / Ollama / local) |
| `genesis/safety.py` | `SafetyCheck` protocol, `AllowAll`, `guarded_chat` (interface only) |
| `genesis/precision.py` | bf16 on CUDA capability >= 8, fp16 on T4, fp32 on CPU |
| `genesis_fin/` | first application: OpenDART finance tasks (depends on core, never the reverse) |
| other `genesis/*` | small-scale pretraining pipeline (research track) |
| `notebooks/fin_colab.ipynb` | Colab runbook: baseline -> QLoRA -> compare |

Data paths (`data/applied/fin_v1`, `runs/applied`, `reports/applied`) are kept for cache compatibility.

## Hard rules (boundaries)

1. **Dependency direction is one way:** core -> `genesis_fin`/private repos. Core must not import `genesis_fin` (enforced by `tests/test_boundaries.py`).
2. **This repo is public.** Never add: company internal data, company names or code, secrets/tokens, or anything about the owner's private *companion* project (safety-pipeline implementations, conversation design, patent candidates). If you see such content, stop and tell the owner.
3. **No new GPU training runs, large downloads, or data additions without the owner's go-ahead.** Document first, then ask.
4. **Never push to `main` directly.** Open a PR; the owner merges (routine PRs may be merged after tests pass if the owner has said so in the current session). The `v0.2.0` tag is created by the owner through GitHub Releases.
5. Do not paste or commit the owner's email or other personal data; commits use the GitHub noreply address.

## Commands

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[fin,qlora]"    # finance task building + QLoRA (GPU machines)
python -m pytest -q              # 93 tests must pass; no GPU needed
```

## Conventions

- Python 3.10+, small modules, type hints where cheap, tests for every behavior change.
- Heavy imports (`torch`, `transformers`) stay lazy so tests and `--help` run on CPU.
- Evaluation numbers must come with: item count, per-kind accuracy, contamination overlap, seconds/item and cost/1k items (`genesis.harness.evaluate` already does this). Do not report a score without the report JSON.
- Prefer accuracy over volume: do less, but correct. State what you could not verify.
- Talk to the owner in Korean, briefly and concretely; no jargon dumps.

## Current state (2026-11-04, verify against `git log` and `docs/roadmap.md`)

- main: structure v0.2.0 merged (PRs #6-#10); `v0.2.0` tag pending (owner).
- fin-v1 baseline (Qwen3-4B, 544 test items): **85.3%** (debt_ratio 61.8, equity_ratio 66.2, net_margin 80.9, operating_margin 91.2, revenue_growth 86.8, roe 95.6, extract 100). The stop criterion "85% and +15 points over baseline" is unreachable with this baseline; it must be redefined (harder eval or a different metric) before more fin rounds.
- First QLoRA was running on a Colab T4 (470 steps, ~2h10m); results go to Google Drive `MyDrive/genesis/results/`.
- Direction from the owner: **do not dig only into finance.** Prefer core work (remaining Gate C items, Gate 1/2 items doable without a GPU) and a second domain over more fin tuning.

## Not doable from a coding agent

Driving Colab / the browser, publishing dashboards, and scheduled collection tasks are done in the Claude app. The owner's personal-project notes live in a Claude Project; ask the owner to paste anything you need.
