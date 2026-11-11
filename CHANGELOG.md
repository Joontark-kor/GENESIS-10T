# Changelog

## Unreleased

- 2026-11-08/09 (polarbear, measured on Qwen3-4B, Colab T4, one run per setting): opt-in `--route` (answer plainly first, keep the answer if it contains no worked-out value; otherwise solve with the formula pipeline); opt-in `--unit-hint`, `--exclude-check`, `--distractor-hint`, `--retry-prefill`. Options that measured no gain were removed (`--direction-retry`, `--multistep`, `--prose-retry`, `--lookup-hint`); their measurements stay in docs/12. New code-generated sets: `multi`, `lookup`, `units` (`genesis_calc.build_tasks --set`). New `choice` scorer and `genesis_calc.external` (KMMLU, Korean GSM8K; rows downloaded at run time, never committed). Colab scripts `colab_repeat.sh`, `colab_external.sh`, `colab_overnight.sh`.
- Evidence so far (docs/12, last section): seeds 1-5, one run each: multi-step 78.7% -> 88.8%; everyday 93.1% -> 97.0%; no-calculation set 85.1% -> 79.2% (`--route` 82.2%, overlaps both); KMMLU 430 items 40.2 / 41.4 / 44.4 (within noise); Korean GSM8K full 1319 items 64.1 / 59.7 / 60.7 (polarbear and route are lower than baseline; paired comparison is significant). Decision 2026-11-11: no new option for the GSM8K-ko result; polarbear is scoped to structured calculation questions (docs/12, last section). Served evaluation equals local evaluation on one set (multi seed 4, 240 items: 86.7% local vs 86.2% through `mini_server`; 1 of 240 verdicts differs); `scripts/colab_serving.sh` repeats it.

- Renamed the "gorilla" pipeline to "polarbear" (module `genesis.harness.polarbear`, `--pipeline polarbear`, docs/12_polarbear_ko.md, notebooks/polarbear_check_*.ipynb). Earlier entries below keep the old name.
- gorilla: two escape hatches closed. A formula typed on the `정답:` line is now calculated, and an answer that is not a formula and not a number copied from the question (worked out in the model's head) gets one retry that asks for the formula only. `--no-pipeline-retry` reproduces the earlier numbers; reports add `pipeline_retried` / `pipeline_rescued`. Known limit: an answer that happens to equal a number in the question is treated as a lookup and not retried. Not yet measured on a GPU.

- Colab without Google Drive: `genesis.hubsync` (private Hugging Face repos; background uploader, restore + `--resume`), `fin_colab.ipynb` reads `fin_v1.zip` from a private dataset repo with the Colab Secret `HF_TOKEN` and pushes checkpoints/reports to a private model repo.

- `notebooks/gorilla_check_colab.ipynb`: a Colab notebook that measures baseline vs gorilla on the everyday-numbers set without Google Drive (no permission popup); the finance set is an optional upload. Calculator now reads `N%` inside a formula as N/100.

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
