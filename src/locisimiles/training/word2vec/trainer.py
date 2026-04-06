"""Word2Vec trainer for Burns-style retrieval models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locisimiles.training.artifacts import resolve_model_output_path
from locisimiles.training.base import BaseTrainer, TrainerConfig
from locisimiles.training.preprocess import tokenize_latin_text


@dataclass(frozen=True)
class Word2VecTrainerConfig(TrainerConfig):
    """Configuration specific to Word2Vec training."""

    vector_size: int = 300
    window: int = 5
    min_count: int = 1
    sg: int = 1
    workers: int = 1
    epochs: int = 10
    output_filename: str = "latin_w2v.model"


class Word2VecTrainer(BaseTrainer):
    """Train a local gensim Word2Vec model from ``seg_id,text`` CSV data."""

    def __init__(self, config: Word2VecTrainerConfig):
        super().__init__(config)
        self.model: Any | None = None

    @property
    def cfg(self) -> Word2VecTrainerConfig:
        return self.config  # type: ignore[return-value]

    def _load_sentences(self) -> list[list[str]]:
        sentences: list[list[str]] = []
        with self.cfg.train_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "text" not in (reader.fieldnames or []):
                raise ValueError("Training CSV must include a 'text' column")

            for row in reader:
                text = row.get("text", "")
                tokens = tokenize_latin_text(
                    text,
                    lowercase=self.cfg.lowercase,
                    normalize_ij_uv=self.cfg.normalize_ij_uv,
                )
                if tokens:
                    sentences.append(tokens)

        if not sentences:
            raise ValueError("No non-empty tokenized training rows found")
        return sentences

    def fit(self, **kwargs: Any) -> Any:
        """Train a gensim Word2Vec model from tokenized training rows."""
        self.validate_data()
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "Word2Vec training requires gensim. Install with: pip install 'locisimiles[word2vec]'"
            ) from exc

        sentences = self._load_sentences()
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            min_count=self.cfg.min_count,
            sg=self.cfg.sg,
            workers=self.cfg.workers,
            seed=self.cfg.seed,
            epochs=self.cfg.epochs,
            **kwargs,
        )
        return self.model

    def save(self, **kwargs: Any) -> Path:
        """Persist the trained model and return its path."""
        if self.model is None:
            raise ValueError("No trained model available. Call fit() first.")
        output_path = resolve_model_output_path(self.cfg.output_dir, self.cfg.output_filename)
        self.model.save(str(output_path))
        return output_path

    def load_artifacts(self, path: str | Path) -> Any:
        """Load an existing gensim Word2Vec model from disk."""
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "Word2Vec training requires gensim. Install with: pip install 'locisimiles[word2vec]'"
            ) from exc

        loaded = Word2Vec.load(str(path))
        self.model = loaded
        return loaded
