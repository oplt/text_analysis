from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MODULES_ROOT = BACKEND_ROOT / "modules"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def _production_python_files(root: Path):
    return (
        path
        for path in root.rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    )


def test_feature_modules_do_not_import_peer_routers() -> None:
    violations: list[str] = []
    for path in _production_python_files(MODULES_ROOT):
        owner = path.relative_to(MODULES_ROOT).parts[0]
        for imported in _imports(path):
            parts = imported.split(".")
            if len(parts) < 4 or parts[:2] != ["backend", "modules"]:
                continue
            imported_owner = parts[2]
            imported_leaf = parts[-1]
            if imported_owner != owner and (
                imported_leaf == "router" or imported_leaf.endswith("_routes")
            ):
                violations.append(f"{path.relative_to(BACKEND_ROOT)} -> {imported}")
    assert not violations, "Peer router imports violate ADR 0003:\n" + "\n".join(violations)


def test_domain_layers_do_not_import_infrastructure() -> None:
    violations = [
        f"{path.relative_to(BACKEND_ROOT)} -> {imported}"
        for path in _production_python_files(MODULES_ROOT)
        if "domain" in path.relative_to(MODULES_ROOT).parts
        for imported in _imports(path)
        if ".infrastructure" in imported
    ]
    assert not violations, "Domain-to-infrastructure imports found:\n" + "\n".join(violations)


def test_observability_does_not_import_feature_modules() -> None:
    observability_root = BACKEND_ROOT / "observability"
    violations = [
        f"{path.relative_to(BACKEND_ROOT)} -> {imported}"
        for path in _production_python_files(observability_root)
        for imported in _imports(path)
        if imported.startswith("backend.modules.")
    ]
    assert not violations, "Observability feature imports found:\n" + "\n".join(violations)
