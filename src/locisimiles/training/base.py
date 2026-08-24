"""Shared training contracts for trainable locisimiles methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class TrainerConfig:
    """Common configuration used by all trainers.

    Trainers take their training data as arguments to ``fit()`` (a
    ``TrainingData``/``Document`` collection, not a config-held path), so
    this base only holds the knobs genuinely common to every trainer.
    """

    output_dir: Path
    seed: int = 42
    lowercase: bool = True
    normalize_ij_uv: bool = True


# =============================================================================
# BaseTrainer
# =============================================================================


class BaseTrainer(ABC):
    """Abstract trainer contract for all trainable methods."""

    def __init__(self, config: TrainerConfig):
        self.config = config

    def validate_data(self) -> None:
        """Validate basic training preconditions common to all trainers.

        Subclasses that take structured data (``Document``/``TrainingData``)
        via ``fit()`` should override this to validate those arguments where
        they're actually available, rather than relying on this base method.
        """
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
