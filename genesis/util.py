"""Small helpers shared by the core and the domain packages."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(sha256: str, val_fraction: float, test_fraction: float) -> str:
    """Deterministic train/val/test bucket from a content hash (a test item never lands in train)."""
    bucket = int(sha256[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"
