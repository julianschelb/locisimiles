"""Text preprocessing helpers shared across trainable methods."""

from __future__ import annotations

import re


def normalize_latin_text(
    text: str,
    *,
    lowercase: bool = True,
    normalize_ij_uv: bool = True,
) -> str:
    """Apply light normalization used by inference-time Word2Vec retrieval."""
    normalized = text
    if lowercase:
        normalized = normalized.lower()
    if normalize_ij_uv:
        normalized = normalized.replace("j", "i").replace("v", "u")
    normalized = re.sub(r"\d+", " ", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_latin_text(
    text: str,
    *,
    lowercase: bool = True,
    normalize_ij_uv: bool = True,
) -> list[str]:
    """Tokenize normalized Latin text into whitespace-separated tokens."""
    normalized = normalize_latin_text(
        text,
        lowercase=lowercase,
        normalize_ij_uv=normalize_ij_uv,
    )
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if token]
