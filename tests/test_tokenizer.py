from pathlib import Path
import unicodedata

import pytest

from genesis.tokenizer import ByteTokenizer, load_tokenizer, train_bpe
from genesis.tokenizer_report import report

SAMPLES = [
    "안녕하세요. 한국어 언어모델을 학습합니다.",
    "트랜스포머는 어텐션과 신경망 연산을 반복합니다. 2026년 10월 29일.",
    "Mixed 한국어 and English, 코드 `print(1+1)` 그리고 이모지 🌍",
    "ㅋㅋㅋ 띄어쓰기가없는문장도처리해야합니다",
]


@pytest.fixture(scope="module")
def bpe(tmp_path_factory) -> Path:
    corpus = SAMPLES * 50 + [f"문서 번호 {i}는 데이터 파이프라인 시험용입니다." for i in range(200)]
    path = tmp_path_factory.mktemp("tok") / "tiny.json"
    train_bpe(iter(corpus), vocab_size=600, output=path, show_progress=False)
    return path


def test_byte_roundtrip():
    text = "안녕하세요. Hello, こんにちは 🌍"
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(text, add_eos=True)) == text
    assert tok.token_byte_lengths().sum() == 256


def test_bpe_is_lossless_and_shorter_than_bytes(bpe: Path):
    tok = load_tokenizer(bpe)
    for text in SAMPLES + ["학습에 없던 낯선 단어: 뷁, 똠방각하"]:
        ids = tok.encode(text, add_eos=True)
        assert ids[-1] == tok.eos_id
        assert tok.decode(ids) == text
    rows = report(SAMPLES, ["byte", str(bpe)])
    assert rows[1]["tokens"] < rows[0]["tokens"] / 2
    assert rows[1]["roundtrip_exact"] == f"{len(SAMPLES)}/{len(SAMPLES)}"


def test_bpe_normalizes_nfd_input(bpe: Path):
    tok = load_tokenizer(bpe)
    nfc = "한국어 정규화"
    nfd = unicodedata.normalize("NFD", nfc)
    assert tok.encode(nfd) == tok.encode(nfc)


def test_bpe_byte_lengths_match_utf8(bpe: Path):
    tok = load_tokenizer(bpe)
    lengths = tok.token_byte_lengths()
    for text in SAMPLES:
        ids = tok.encode(text)
        assert int(lengths[ids].sum()) == len(text.encode("utf-8"))
    assert lengths[tok.eos_id] == 0


def test_tokenizer_hash_is_checked(bpe: Path):
    with pytest.raises(ValueError):
        load_tokenizer(bpe, expected_sha256="0" * 64)
