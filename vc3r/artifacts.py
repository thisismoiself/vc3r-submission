"""Pinned external model identifiers and acquisition helpers."""

from __future__ import annotations

from pathlib import Path


DA3_MODEL_ID = "depth-anything/DA3-LARGE-1.1"
DA3_REVISION = "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"


def resolve_da3_snapshot(
    model_name: str | Path = DA3_MODEL_ID,
    revision: str = DA3_REVISION,
    local_files_only: bool = False,
) -> Path:
    """Resolve an explicit snapshot or an exact Hugging Face model revision."""
    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        snapshot = candidate.resolve()
    elif candidate.exists():
        raise ValueError(f"DA3 model path is not a directory: {candidate}")
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            repo_id=str(model_name),
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=("config.json", "model.safetensors"),
        ))

    required = (snapshot / "config.json", snapshot / "model.safetensors")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete DA3 snapshot; missing: {', '.join(missing)}")
    return snapshot
