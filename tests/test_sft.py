from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from genesis.chat import encode_conversation, generation_prompt, render
from genesis.config import ModelConfig
from genesis.data.sft import SFTSplit, build, to_messages
from genesis.model import GenesisLM
from genesis.tokenizer import load_tokenizer, train_bpe
from genesis.train import main as train_main

CONVERSATIONS = [
    [{"role": "user", "content": f"{i}번 질문: 한국의 수도는?"},
     {"role": "assistant", "content": f"서울입니다. ({i})"}] for i in range(40)
] + [
    [{"role": "system", "content": "친절하게 답하세요."},
     {"role": "user", "content": "1 + 1 은?\n 계산해줘"},
     {"role": "assistant", "content": " 2 입니다.\n\n끝."}],
]


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory) -> Path:
    corpus = [render(c) for c in CONVERSATIONS] * 5 + ["한국어 문장 데이터 " * 5] * 20
    path = tmp_path_factory.mktemp("tok") / "tok.json"
    train_bpe(iter(corpus), vocab_size=500, output=path, show_progress=False)
    return path


def test_mask_covers_only_assistant_content_and_end(tokenizer_path):
    tok = load_tokenizer(tokenizer_path)
    ids, mask = encode_conversation(tok, CONVERSATIONS[-1])
    learned = tok.decode([t for t, m in zip(ids, mask) if m], skip_special=False)
    # the leading whitespace may share a token with the header newline (as in HF tokenization)
    assert learned.lstrip() == "2 입니다.\n\n끝.<|im_end|>"
    assert "assistant" not in learned and "계산" not in learned
    assert tok.decode(ids, skip_special=False) == render(CONVERSATIONS[-1])


def test_encoding_matches_hf_chat_template(tokenizer_path):
    transformers = pytest.importorskip("transformers")
    from genesis.chat import HF_CHAT_TEMPLATE
    hf = transformers.PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    hf.chat_template = HF_CHAT_TEMPLATE
    tok = load_tokenizer(tokenizer_path)
    for conversation in CONVERSATIONS[:2] + CONVERSATIONS[-1:]:
        ours, _ = encode_conversation(tok, conversation)
        theirs = hf.apply_chat_template(conversation, tokenize=True)
        theirs = theirs["input_ids"] if isinstance(theirs, dict) or hasattr(theirs, "keys") else theirs
        assert ours == list(theirs)
        prompt = generation_prompt(tok, conversation[:-1])
        theirs_prompt = hf.apply_chat_template(conversation[:-1], tokenize=True, add_generation_prompt=True)
        theirs_prompt = theirs_prompt["input_ids"] if hasattr(theirs_prompt, "keys") else theirs_prompt
        assert prompt == list(theirs_prompt)


def test_alpaca_records_are_converted():
    messages = to_messages({"instruction": "번역해줘", "input": "hello", "output": "안녕"})
    assert messages == [{"role": "user", "content": "번역해줘\n\nhello"}, {"role": "assistant", "content": "안녕"}]


def _write_jsonl(path: Path) -> Path:
    lines = [json.dumps({"messages": c}, ensure_ascii=False) for c in CONVERSATIONS]
    lines += [lines[0], "not json", json.dumps({"messages": [{"role": "user", "content": "답 없음"}]}),
              json.dumps({"messages": [{"role": "user", "content": "연락처 010-1234-5678"},
                                       {"role": "assistant", "content": "확인했습니다."}]}, ensure_ascii=False)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_build_dedups_masks_pii_and_splits(tmp_path, tokenizer_path):
    src = _write_jsonl(tmp_path / "sft.jsonl")
    manifest = build([src], str(tokenizer_path), tmp_path / "sft", source="unit", license_id="self-authored",
                     rights_basis="self-authored", val_fraction=0.2)
    stats = manifest["statistics"]
    assert stats["exact_duplicate"] == 1 and stats["invalid"] == 2 and stats["kept"] == 42
    assert manifest["pii_masked"] == {"phone": 1}
    assert manifest["splits"]["train"]["examples"] + manifest["splits"]["val"]["examples"] == 42
    split = SFTSplit(tmp_path / "sft", "train")
    x, y = split.sample(np.random.default_rng(0), 4, seq_len=64)
    assert x.shape == y.shape and (y != -100).any() and (y == -100).any()
    with pytest.raises(ValueError):
        build([src], str(tokenizer_path), tmp_path / "x", source="unit", license_id="unknown",
              rights_basis="self-authored")


def test_sft_training_from_pretrained_weights(tmp_path, tokenizer_path):
    src = _write_jsonl(tmp_path / "sft.jsonl")
    build([src], str(tokenizer_path), tmp_path / "sft", source="unit", license_id="self-authored",
          rights_basis="self-authored", val_fraction=0.2, max_length=65)
    tok = load_tokenizer(tokenizer_path)
    config = ModelConfig(vocab_size=tok.vocab_size, context_length=64, n_layers=2, n_heads=4, d_model=32,
                         n_experts=1, experts_per_token=1, expert_hidden_dim=64)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(asdict(config)))
    torch.manual_seed(0)
    base = tmp_path / "base.pt"
    torch.save({"model_state": GenesisLM(config).state_dict(), "config": asdict(config),
                "tokenizer": tok.describe(), "step": 0}, base)
    args = ["--config", str(cfg_path), "--data", str(tmp_path / "sft"), "--out-dir", str(tmp_path / "run"),
            "--steps", "30", "--batch-size", "8", "--lr", "5e-3", "--eval-every", "10", "--device", "cpu",
            "--init-from", str(base)]
    result = train_main(args)
    evals = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
             if json.loads(l)["type"] == "eval"]
    assert evals[-1]["loss"] < evals[0]["loss"]
    assert result["val"]["tokens"] > 0
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["stage"] == "sft" and run["init_from"] == str(base)

    too_short = ModelConfig(**{**asdict(config), "context_length": 16})
    cfg_path.write_text(json.dumps(asdict(too_short)))
    with pytest.raises(ValueError):
        train_main(args[:-2])
