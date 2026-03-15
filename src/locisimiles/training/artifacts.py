"""Artifact path helpers for training workflows."""

from __future__ import annotations

from pathlib import Path


def resolve_model_output_path(output_dir: str | Path, filename: str) -> Path:
    """Return a normalized output path and ensure parent directory exists."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename
