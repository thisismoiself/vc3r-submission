#!/usr/bin/env python3
"""Command-line entrypoint for the Replica office4 evaluator."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vc3r_eval.office4 import main  # noqa: E402


if __name__ == "__main__":
    main()
