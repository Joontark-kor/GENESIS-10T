import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

from genesis.applied.build_fin_tasks import build
from genesis.applied.chatdata import encode_example
from genesis.applied.evaluate_tasks import is_correct, main as eval_main, parse_number
from genesis.applied.finetune import main as finetune_main
from genesis.chat import HF_CHAT_TEMPLATE
from genesis.tokenizer import train_bpe
from tests.test_applied_fin import SAMSUNG_LIKE, _record


def test_parse_number_prefers_tagged_answer():
    assert parse_number("계산하면 1,234 이고\n정답: 2.54%") == 2.54
    assert parse_number("答え: -12.5%") == -12.5
    assert parse_number("결과는 △3.10% 입니다") == -3.10
    assert parse_number("숫자 없음") is None
    assert parse_number("1 2 3 마지막 42.0") == 42.0
    assert is_correct(2.545, 2.54, 0.01) and not is_correct(2.6, 2.54, 0.01) and not is_correct(None, 1, 1)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> dict:
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("applied")
    out = root / "fin"
    out.mkdir()
    with open(out / "records.jsonl", "w", encoding="utf-8") as handle:
        for i in range(30):
            handle.write(json.dumps(_record(code=f"{i:08d}", name=f"회사{i}", year=2025), ensure_ascii=False) + "\n")
    build(out, ["ko"], val_fraction=0.2, test_fraction=0.2)

    corpus = [line for name in ("sft_train.jsonl", "sft_val.jsonl")
              for line in (out / name).read_text(encoding="utf-8").splitlines()]
    texts = [m["content"] for line in corpus for m in json.loads(line)["messages"]]
    tok_json = root / "tok.json"
    train_bpe(iter(texts * 2 + ["system user assistant"] * 10), vocab_size=700, output=tok_json, show_progress=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_json), eos_token="<|im_end|>", pad_token="<|pad|>")
    tokenizer.chat_template = HF_CHAT_TEMPLATE
    model_dir = root / "base"
    tokenizer.save_pretrained(model_dir)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                         bos_token_id=None, tie_word_embeddings=True)
    LlamaForCausalLM(config).save_pretrained(model_dir)
    return {"root": root, "data": out, "model": model_dir, "tokenizer": tokenizer}


def test_assistant_only_labels(workspace):
    tok = workspace["tokenizer"]
    row = json.loads((workspace["data"] / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    item = encode_example(tok, row["messages"], max_length=2048)
    learned = tok.decode([t for t, l in zip(item["input_ids"], item["labels"]) if l != -100])
    answer = row["messages"][-1]["content"]
    assert learned.replace(" ", "").startswith(answer.replace(" ", "")[:20])
    assert learned.endswith("<|im_end|>")
    assert "질문" not in learned and "분석 보조" not in learned
    assert encode_example(tok, row["messages"], max_length=10) is None


def test_finetune_then_evaluate(workspace):
    out = workspace["root"] / "run"
    run = finetune_main(["--base-model", str(workspace["model"]), "--train", str(workspace["data"] / "sft_train.jsonl"),
                         "--val", str(workspace["data"] / "sft_val.jsonl"), "--output", str(out), "--cpu",
                         "--max-steps", "4", "--batch-size", "2", "--grad-accum", "1", "--lr", "1e-3",
                         "--eval-steps", "2", "--logging-steps", "1", "--lora-r", "4", "--lora-alpha", "8"])
    assert (out / "adapter" / "adapter_config.json").exists()
    assert run["steps"] == 4 and run["trainable_parameters"] < run["total_parameters"]
    assert run["data"]["train"]["used"] > 0 and run["data"]["train"]["skipped"] == 0
    saved = json.loads((out / "run.json").read_text())
    assert saved["method"] == "LoRA" and saved["data"]["train"]["sha256"]

    report_base = workspace["root"] / "base.json"
    report_tuned = workspace["root"] / "tuned.json"
    common = ["--data", str(workspace["data"] / "eval_test.jsonl"), "--limit", "4", "--max-new-tokens", "8",
              "--batch-size", "2", "--cpu", "--gpu-hour-price", "2.5"]
    base = eval_main(["run", "--model", str(workspace["model"]), "--name", "base", "--out", str(report_base), *common])
    tuned = eval_main(["run", "--model", str(workspace["model"]), "--adapter", str(out / "adapter"),
                       "--name", "tuned", "--out", str(report_tuned), *common])
    for report in (base, tuned):
        assert report["items"] == 4 and 0.0 <= report["accuracy"] <= 1.0
        assert report["speed"]["generated_tokens"] > 0 and report["cost"]["gpu_hours"] == 0.0
    compared = eval_main(["compare", str(report_base), str(report_tuned)])
    assert [r["name"] for r in compared] == ["base", "tuned"]
