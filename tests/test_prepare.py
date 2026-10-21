from pathlib import Path
import numpy as np
from genesis.prepare import prepare


def test_prepare_dedup_and_splits(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("A sentence from a licensed corpus. " * 15, encoding="utf-8")
    (raw / "b.txt").write_text("A sentence from a licensed corpus. " * 15, encoding="utf-8")
    (raw / "c.txt").write_text("A different text to hold out. " * 15, encoding="utf-8")
    out = tmp_path / "processed"
    manifest = prepare(raw, out, source="test", license_id="self-authored")
    assert manifest["unique_documents"] == 2
    assert len(np.fromfile(out / "train.bin", dtype=np.uint16)) > 0
    assert len(np.fromfile(out / "val.bin", dtype=np.uint16)) > 0
