#!/usr/bin/env python3
"""Download the exact DA3 snapshot used by the VC3R evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download  # noqa: E402
from vc3r.artifacts import DA3_MODEL_ID, DA3_REVISION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "checkpoints/da3/DA3-LARGE-1.1",
        help="destination directory for the pinned snapshot",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    snapshot_download(
        repo_id=DA3_MODEL_ID,
        revision=DA3_REVISION,
        local_dir=output,
        allow_patterns=("config.json", "model.safetensors", "README.md"),
    )
    print(f"Downloaded {DA3_MODEL_ID}@{DA3_REVISION} to {output}")


if __name__ == "__main__":
    main()
