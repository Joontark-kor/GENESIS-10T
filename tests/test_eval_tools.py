import json
from pathlib import Path

from genesis.contamination import ContaminationIndex, check, normalize_for_match
from genesis.data.clean import clean, iter_clean
from genesis.data.sources import SourceSpec

LEAKED = "다음 중 대한민국의 수도는 어디인가요 서울 부산 대구 광주 중에서 고르시오"


def _clean_corpus(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    docs = [
        {"id": "a", "text": "오늘은 데이터 정제 파이프라인을 설명합니다. " * 3},
        {"id": "b", "text": f"시험 문제 유출 예시입니다. {LEAKED}. 정답은 서울입니다."},
        {"id": "c", "text": "언어모델 평가는 오염 검사와 함께 보고해야 신뢰할 수 있습니다."},
    ]
    (raw / "d.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    spec = SourceSpec(name="t", type="jsonl", path=str(raw), license="self-authored",
                      rights_basis="self-authored", id_field="id", filters={"min_chars": 10})
    out = tmp_path / "clean"
    clean([spec], out, val_fraction=0.0, test_fraction=0.0, near_dedup=False)
    return out


def test_normalization_ignores_punctuation_and_case():
    assert normalize_for_match("Hello,  World! 서울?") == "hello world 서울"


def test_contamination_finds_leaked_item_and_blocklist_removes_it(tmp_path):
    clean_dir = _clean_corpus(tmp_path)
    items = [LEAKED, "이 문항은 학습 데이터 어디에도 등장하지 않는 완전히 새로운 질문입니다 정말로"]
    report, bad_docs = check(clean_dir, items, n=8)
    assert report["contaminated_items"] == 1 and report["examples"][0]["item_index"] == 0
    assert len(bad_docs) == 1

    rebuilt = tmp_path / "clean2"
    spec = SourceSpec(name="t", type="jsonl", path=str(tmp_path / "raw"), license="self-authored",
                      rights_basis="self-authored", id_field="id", filters={"min_chars": 10})
    manifest = clean([spec], rebuilt, val_fraction=0.0, test_fraction=0.0, near_dedup=False,
                     exclude_sha256=bad_docs)
    assert manifest["statistics"]["t"]["removed"]["blocklisted"] == 1
    assert check(rebuilt, items, n=8)[0]["contaminated_items"] == 0
    assert len(list(iter_clean(rebuilt, "train"))) == 2


def test_short_items_match_as_whole_phrases():
    index = ContaminationIndex(["서울 부산"], n=8)
    assert index.match("여행지는 서울 부산 순서") == {0}
    assert index.match("서울부산") == set()
