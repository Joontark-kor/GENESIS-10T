"""Sealed test sets: publish only a hash, open once.

A sealed set is a JSONL file kept OUTSIDE this public repo (private repo or local disk). This repo
holds only its manifest: SHA-256 of the file, item count, creation date and the frozen scorer
version. ``open_sealed`` refuses a file whose hash differs (edited after sealing) and records the
opening in a log so a second look is visible.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(path: str | Path, name: str, scorer_version: str, manifest_dir: str | Path) -> dict:
    path = Path(path)
    items = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    manifest = {"name": name, "sha256": sha256_file(path), "items": items,
                "sealed_at": dt.date.today().isoformat(), "scorer_version": scorer_version, "opened": []}
    out = Path(manifest_dir) / f"{name}.sealed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def open_sealed(path: str | Path, manifest_path: str | Path, purpose: str) -> list[dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["sha256"]:
        raise ValueError(f"{manifest['name']}: file hash differs from the sealed manifest")
    if manifest["opened"]:
        raise PermissionError(f"{manifest['name']} was already opened on {manifest['opened'][0]['at']}; "
                              "make a new sealed set instead of reusing this one")
    manifest["opened"].append({"at": dt.datetime.now().isoformat(timespec="minutes"), "purpose": purpose})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
