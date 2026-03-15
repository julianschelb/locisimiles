"""Shared training contracts for trainable locisimiles methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    """Common configuration used by all trainers."""

    train_path: Path
    output_dir: Path
    seed: int = 42
    lowercase: bool = True
    normalize_ij_uv: bool = True


class BaseTrainer(ABC):
    """Abstract trainer contract for all trainable methods."""

    def __init__(self, config: TrainerConfig):
        self.config = config

    def validate_data(self) -> None:
        """Validate input paths and basic training preconditions."""
        if not self.config.train_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.config.train_path}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fit(self, **kwargs: Any) -> Any:
        """Train model artifacts from input data."""
        ...

    @abstractmethod
    def save(self, **kwargs: Any) -> Path:
        """Persist trained artifacts and return the primary output path."""
        ...

    @abstractmethod
    def load_artifacts(self, path: str | Path) -> Any:
        """Load persisted artifacts for inspection or reuse."""
        ...
