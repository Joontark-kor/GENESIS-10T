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


def test_core_has_no_old_applied_package():
    # git pull leaves an empty genesis/applied/__pycache__ behind on old checkouts; only source files count
    assert not list((ROOT / "genesis" / "applied").glob("*.py"))
