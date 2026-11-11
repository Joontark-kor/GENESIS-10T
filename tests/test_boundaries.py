"""Repository boundaries: dependencies flow one way, from the GENESIS core outward."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imports(package: str) -> dict[str, set[str]]:
    found = {}
    for path in (ROOT / package).rglob("*.py"):
        names = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
        found[str(path.relative_to(ROOT))] = names
    return found


def test_core_never_imports_domain_packages():
    domain = {"genesis_fin", "genesis_calc", "genesis_predict"}
    offenders = {f: n & domain for f, n in _imports("genesis").items() if n & domain}
    assert not offenders, f"genesis core must not depend on domain packages: {offenders}"


def test_nothing_here_imports_the_pretraining_repo():
    # Track B (from-scratch pretraining) moved to the separate `ssial` repository on 2026-11-11
    offenders = {f: n & {"ssial"} for pkg in ("genesis", "genesis_fin", "genesis_calc", "genesis_predict")
                 for f, n in _imports(pkg).items() if n & {"ssial"}}
    assert not offenders, f"GENESIS-10T must not depend on ssial: {offenders}"


def test_pretraining_modules_are_gone():
    moved = ["model", "train", "tokenizer", "config", "chat", "estimate", "ladder", "prepare", "text", "generate"]
    assert not [m for m in moved if (ROOT / "genesis" / f"{m}.py").exists()]
    assert not (ROOT / "genesis" / "data").exists()


def test_core_has_no_old_applied_package():
    # git pull leaves an empty genesis/applied/__pycache__ behind on old checkouts; only source files count
    assert not list((ROOT / "genesis" / "applied").glob("*.py"))
