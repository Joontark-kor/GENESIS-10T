import gzip
import json
from pathlib import Path
import unicodedata

import numpy as np
import pytest

from genesis.data.clean import assign_split, clean, iter_clean
from genesis.data.dedup import MinHashDeduper, exact_key
from genesis.data.filters import QualityConfig, quality_reason
from genesis.data.pii import mask_pii
from genesis.data.shards import ShardWriter, TokenSplit
from genesis.data.sources import SourceSpec, load_registry
from genesis.data.tokenize import tokenize_dataset
from genesis.prepare import prepare
from genesis.text import hangul_ratio, normalize_text
from genesis.tokenizer import token_dtype

KOREAN_DOC = ("한국어 언어모델을 처음부터 학습하려면 데이터의 출처와 권리를 먼저 확인해야 합니다. "
              "그다음 정규화, 품질 필터, 개인정보 마스킹, 중복 제거를 거쳐 문서 단위로 분할합니다. ")


# --- text -------------------------------------------------------------------

def test_normalize_text_converts_nfd_hangul_to_nfc():
    nfd = unicodedata.normalize("NFD", "한국어\r\n문장​")
    assert normalize_text(nfd) == "한국어\n문장"
    assert len(normalize_text(nfd)) == 6


def test_hangul_ratio_ignores_digits_and_punctuation():
    assert hangul_ratio("한국 2026!") == 1.0
    assert hangul_ratio("abc 가나") == pytest.approx(0.4)


# --- filters, PII, dedup ----------------------------------------------------

def test_quality_filters_report_reasons():
    q = QualityConfig()
    assert quality_reason("짧다", "ko", q) == "too_short"
    english = ("Language models learn statistical patterns from large text corpora. "
               "Careful data curation matters as much as architecture and compute budget.")
    assert quality_reason(english, "ko", q) == "low_hangul_ratio"
    assert quality_reason(english, "en", q) is None
    assert quality_reason("This is spam text. " * 20, "en", q) == "repetitive_ngrams"
    assert quality_reason("홈\n메뉴\n로그인\n가입\n검색\n" * 6, "ko", q.with_overrides({"min_chars": 10})) in {
        "duplicate_lines", "short_lines"}
    assert quality_reason(KOREAN_DOC * 2, "ko", q) is None
    with pytest.raises(ValueError):
        q.with_overrides({"no_such_rule": 1})


def test_pii_masking_korean_identifiers():
    text = ("연락처 010-1234-5678, 사무실 02-123-4567, 메일 hong.gildong@example.co.kr, "
            "주민번호 900101-1234567, 카드 4111 1111 1111 1111, 연도 2026-10-29")
    masked, counts = mask_pii(text)
    assert "010-1234-5678" not in masked and "02-123-4567" not in masked
    assert "example.co.kr" not in masked and "900101-1234567" not in masked
    assert "4111" not in masked
    assert "2026-10-29" in masked  # dates are not phone numbers
    assert counts == {"rrn": 1, "phone": 2, "email": 1, "card": 1}


def test_card_requires_luhn():
    masked, counts = mask_pii("주문번호 1234 5678 9012 3456")
    assert counts["card"] == 0 and "1234 5678 9012 3456" in masked


def test_minhash_finds_near_duplicates_only():
    dedup = MinHashDeduper(threshold=0.8)
    base = KOREAN_DOC * 3
    assert not dedup.is_duplicate(base)
    assert dedup.is_duplicate(base.replace("확인해야", "확인하여야", 1))
    assert not dedup.is_duplicate("전혀 다른 주제의 문서입니다. 날씨와 여행 이야기를 담고 있습니다. " * 4)
    assert exact_key("가  나\n다") == exact_key("가 나 다")


# --- registry ---------------------------------------------------------------

def test_registry_requires_license_and_rights(tmp_path: Path):
    with pytest.raises(ValueError):
        SourceSpec(name="x", type="txt", path=".", license="unknown", rights_basis="open-license")
    with pytest.raises(ValueError):
        SourceSpec(name="x", type="txt", path=".", license="CC-BY-4.0", rights_basis="probably-fine")
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps({"sources": [
        {"name": "a", "type": "txt", "path": "raw", "license": "self-authored", "rights_basis": "self-authored"},
        {"name": "a", "type": "txt", "path": "raw", "license": "self-authored", "rights_basis": "self-authored"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(registry)


def test_example_registry_is_valid():
    specs = load_registry(Path(__file__).parent.parent / "configs" / "sources.example.json")
    assert len(specs) >= 3


# --- end to end -------------------------------------------------------------

def _make_corpus(tmp_path: Path, n_docs: int = 60) -> SourceSpec:
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = []
    for i in range(n_docs):
        lines.append(json.dumps({"id": i, "text": f"{i}번째 문서입니다. 주제 번호 {i * 7919 % 1000}. " + KOREAN_DOC}, ensure_ascii=False))
    lines.append(json.dumps({"id": "dup", "text": f"0번째 문서입니다. 주제 번호 0. " + KOREAN_DOC}, ensure_ascii=False))
    lines.append(json.dumps({"id": "en", "text": "English only text that should be filtered out. " * 5}))
    with gzip.open(raw / "docs.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return SourceSpec(name="unit", type="jsonl", path=str(raw), license="self-authored",
                      rights_basis="self-authored", id_field="id", filters={"min_chars": 20})


def test_clean_then_tokenize_end_to_end(tmp_path: Path):
    spec = _make_corpus(tmp_path)
    clean_dir = tmp_path / "clean"
    manifest = clean([spec], clean_dir, val_fraction=0.15, test_fraction=0.15, near_dedup=False)
    stats = manifest["statistics"]["unit"]
    assert stats["seen"] == 62 and stats["kept"] == 60
    assert stats["removed"] == {"exact_duplicate": 1, "low_hangul_ratio": 1}
    ids = {split: {r["sha256"] for r in iter_clean(clean_dir, split)} for split in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert all(ids.values()), "every split should receive documents"
    assert (clean_dir / "DATA_CARD.md").read_text(encoding="utf-8").count("unit") >= 2

    out = tmp_path / "tokens"
    token_manifest = tokenize_dataset(clean_dir, "byte", out, max_tokens_per_shard=2000)
    assert token_manifest["split_strategy"].startswith("document-held-out")
    assert len(token_manifest["splits"]["train"]["shards"]) > 1
    train = TokenSplit(out, "train")
    assert train.total == token_manifest["splits"]["train"]["tokens"]
    x, y = train.sample(np.random.default_rng(0), batch_size=3, seq_len=32)
    assert x.shape == (3, 32) and np.array_equal(x[:, 1:], y[:, :-1])
    windows = list(TokenSplit(out, "val").windows(seq_len=32))
    assert windows and all(len(a) == 32 for a, _ in windows)


def test_split_assignment_is_stable():
    sha = "00000000" + "f" * 56
    assert assign_split(sha, 0.1, 0.1) == "test"
    assert assign_split("ffffffff" + "0" * 56, 0.1, 0.1) == "train"


def test_shard_writer_rejects_out_of_range_ids(tmp_path: Path):
    writer = ShardWriter(tmp_path, "train", np.uint16)
    with pytest.raises(ValueError):
        writer.write([70000])
    writer.close()
    assert token_dtype(257) == np.uint16
    assert token_dtype(65536) == np.uint16
    assert token_dtype(131072) == np.uint32


def test_prepare_tiny_corpus_discloses_weak_split(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("A sentence from a licensed corpus. " * 15, encoding="utf-8")
    (raw / "b.txt").write_text("A sentence from a licensed corpus. " * 15, encoding="utf-8")
    (raw / "c.txt").write_text("A different text to hold out. " * 15, encoding="utf-8")
    manifest = prepare(raw, tmp_path / "processed", source="test", license_id="self-authored")
    assert manifest["unique_documents"] == 2
    for split in ("train", "val", "test"):
        assert manifest["splits"][split]["tokens"] > 0
    with pytest.raises(ValueError):
        prepare(raw, tmp_path / "other", source="test", license_id=" ")
