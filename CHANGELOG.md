# Changelog

## Unreleased

- Colab notebook: remove Colab's preinstalled `torchao` (an old version made peft fail with `ImportError` when loading an adapter), write the run directory straight to Drive (a disconnected session keeps its checkpoints), and copy only `reports/` in the save cell.

## v0.2.0 — 2026-11 — the GENESIS core as a contract

Breaking: `genesis.applied` is gone. New locations:

| Before | Now |
|---|---|
| `genesis.applied.dart` | `genesis_fin.dart` |
| `genesis.applied.build_fin_tasks` | `genesis_fin.build_tasks` |
| `genesis.applied.chatdata`, `finetune` | `genesis.adapt.chatdata`, `genesis.adapt.finetune` |
| `genesis.applied.evaluate_tasks` | `genesis.harness.evaluate` (scorers in `genesis.harness.scorers`) |
| `pip install -e ".[applied]"` | `pip install -e ".[fin]"` (`applied` still works) |
| `notebooks/applied_colab.ipynb`, `docs/10_applied_ko.md` | `notebooks/fin_colab.ipynb`, `docs/10_fin_ko.md` |

Data, run and report paths (`data/applied/`, `runs/applied/`, `reports/applied/`) are unchanged, so existing DART caches and task files keep working.

Added:
- Evaluation reports carry `report_version`, `genesis_version` and `contamination` (exact prompt overlap with `--train-data`); pluggable scorers (`numeric`, `exact`, `module:function`).
- `genesis.harness.evaluate run --endpoint URL` evaluates a served model the way apps call it.
- `genesis.serving`: OpenAI-compatible endpoint client (standard library only) and the vLLM multi-LoRA command.
- `genesis.safety`: the safety-check interface (`check_input`, `check_output`, `Verdict`, `guarded_chat`, `load_safety_check`). The core ships no rules; without a configured check, `AllowAll` reports that no check ran.
- `tests/test_boundaries.py`: the core never imports domain packages.
- `docs/11_core_ko.md`: the three layers and the contract.

## v0.1 — 2026-10/11

Gates 0–2 tooling, ko32k-v1 tokenizer, GENESIS-Applied (OpenDART tasks, LoRA/QLoRA, evaluator), SPAC exclusion and the Colab notebook.
