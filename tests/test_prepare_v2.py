import pytest

from genesis.prepare_v2 import split_documents_v2


def test_split_ratio():
    documents = list(range(20))
    result = split_documents_v2(documents, seed=42)

    assert len(result["train"]) == 16
    assert len(result["validation"]) == 2
    assert len(result["test"]) == 2


def test_no_overlap():
    documents = list(range(20))
    result = split_documents_v2(documents, seed=42)

    train = set(result["train"])
    validation = set(result["validation"])
    test = set(result["test"])

    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)


def test_insufficient_documents():
    with pytest.raises(ValueError):
        split_documents_v2(list(range(5)))

import json
import numpy as np

from genesis.prepare_v2 import save_splits_v2


def test_save_splits(tmp_path):
    documents = [
        {"text": f"GENESIS document {i}"}
        for i in range(20)
    ]

    splits = split_documents_v2(documents, seed=42)
    metadata = save_splits_v2(splits, tmp_path)

    # Check that all expected files exist.
    for filename in [
        "train.bin",
        "val.bin",
        "test.bin",
        "manifest.json",
    ]:
        assert (tmp_path / filename).exists()

    # Check that binary files contain tokens.
    for filename in ["train.bin", "val.bin", "test.bin"]:
        tokens = np.fromfile(tmp_path / filename, dtype=np.uint16)
        assert len(tokens) > 0

    # Check document counts.
    assert metadata["splits"]["train"]["documents"] == 16
    assert metadata["splits"]["validation"]["documents"] == 2
    assert metadata["splits"]["test"]["documents"] == 2

    # Check saved metadata.
    saved = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert saved == metadata

from datetime import datetime


def test_metadata_provenance(tmp_path):
    documents = [
        {"text": f"GENESIS document {i}"}
        for i in range(20)
    ]

    splits = split_documents_v2(documents, seed=42)

    metadata = save_splits_v2(
        splits,
        tmp_path,
        source="GENESIS-10T",
        license_id="self-authored",
    )

    assert metadata["source"] == "GENESIS-10T"
    assert metadata["license"] == "self-authored"

    # Check that the timestamp is valid.
    datetime.fromisoformat(metadata["created_at"])

    # Check that the information was saved to disk.
    saved = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert saved["source"] == "GENESIS-10T"
    assert saved["license"] == "self-authored"
    assert saved["created_at"] == metadata["created_at"]