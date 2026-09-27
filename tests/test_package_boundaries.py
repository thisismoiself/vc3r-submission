"""Dependency-free checks for the VC3R package ownership boundary."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackageBoundaryTest(unittest.TestCase):
    def test_top_level_packages_import_without_eager_model_dependencies(self) -> None:
        importlib.import_module("vc3r")
        importlib.import_module("vc3r_eval")

    def test_vc3r_does_not_import_vc3r_eval(self) -> None:
        violations: list[str] = []
        for path in (REPO_ROOT / "vc3r").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(name == "vc3r_eval" or name.startswith("vc3r_eval.") for name in names):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
