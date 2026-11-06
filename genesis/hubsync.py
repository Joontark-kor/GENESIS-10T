"""Keep run folders and data on private Hugging Face repos so a Colab session can be lost and restarted
with no click on a Google permission popup.

The token is never written to a file or printed: callers pass it in (Colab: ``userdata.get("HF_TOKEN")``).

    sync_down(repo, local)   # restore a run folder (checkpoints) if the repo already has one
    Uploader(repo, local).start()   # background thread: push the folder every few minutes
    fetch_file(repo, name, out)     # one file from a private dataset repo, e.g. fin_v1.zip
"""
from __future__ import annotations

import threading
from pathlib import Path


def _api(token: str):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def username(token: str) -> str:
    return _api(token).whoami()["name"]


def ensure_repo(repo: str, token: str, repo_type: str = "model") -> None:
    """Create the repo as private when it does not exist (never changes an existing repo's visibility)."""
    _api(token).create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)


def sync_down(repo: str, local: Path | str, token: str, repo_type: str = "model") -> bool:
    """Download the repo into ``local``. Returns False when the repo has nothing yet."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        snapshot_download(repo, repo_type=repo_type, local_dir=str(local), token=token)
    except RepositoryNotFoundError:
        return False
    return any(Path(local).iterdir()) if Path(local).exists() else False


def sync_up(repo: str, local: Path | str, token: str, repo_type: str = "model", path_in_repo: str | None = None) -> None:
    """Upload ``local`` (only changed files are sent). Repo is created private if missing."""
    ensure_repo(repo, token, repo_type)
    _api(token).upload_folder(repo_id=repo, repo_type=repo_type, folder_path=str(local),
                              path_in_repo=path_in_repo, commit_message="sync from Colab")


def fetch_file(repo: str, filename: str, out_dir: Path | str, token: str, repo_type: str = "dataset") -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type=repo_type, local_dir=str(out_dir), token=token))


class Uploader(threading.Thread):
    """Push ``local`` to ``repo`` every ``every`` seconds until stopped; one failed push does not stop it."""

    def __init__(self, repo: str, local: Path | str, token: str, every: int = 600, push=sync_up):
        super().__init__(daemon=True)
        self.repo, self.local, self.token, self.every, self._push = repo, Path(local), token, every, push
        self._stop_event = threading.Event()
        self.pushes = self.failures = 0
        self.last_error = ""

    def push_once(self) -> bool:
        if not self.local.exists() or not any(self.local.iterdir()):
            return False
        try:
            self._push(self.repo, self.local, self.token)
        except Exception as exc:  # network blip, rate limit: try again next round
            self.failures += 1
            self.last_error = type(exc).__name__  # never the message: it can echo request details
            return False
        self.pushes += 1
        return True

    def run(self) -> None:
        while not self._stop_event.wait(self.every):
            self.push_once()

    def stop(self, final_push: bool = True) -> None:
        self._stop_event.set()
        if final_push:
            self.push_once()
