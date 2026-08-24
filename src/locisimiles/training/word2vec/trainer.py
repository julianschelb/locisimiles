"""Word2Vec trainer for Burns-style retrieval models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from locisimiles.document import Document
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
    """Train a local gensim Word2Vec model from one or more ``Document``s.

    Word2Vec is unsupervised (it learns word embeddings from raw sentences,
    not labeled pairs), so unlike the pair/label trainers it takes plain
    ``Document``s rather than a ``TrainingData``.
    """

    def __init__(self, config: Word2VecTrainerConfig):
        super().__init__(config)
        self.model: Any | None = None

    @property
    def cfg(self) -> Word2VecTrainerConfig:
        return self.config  # type: ignore[return-value]

    def validate_data(self) -> None:
        """Ensure the output directory exists; ``fit()`` validates its documents."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_sentences(self, documents: Sequence[Document]) -> list[list[str]]:
        sentences: list[list[str]] = []
        for document in documents:
            for segment in document:
                tokens = tokenize_latin_text(
                    segment.text,
                    lowercase=self.cfg.lowercase,
                    normalize_ij_uv=self.cfg.normalize_ij_uv,
                )
                if tokens:
                    sentences.append(tokens)

        if not sentences:
            raise ValueError("No non-empty tokenized training rows found")
        return sentences

    def fit(self, *, documents: Sequence[Document], **kwargs: Any) -> Any:  # type: ignore[override]
        """Train a gensim Word2Vec model from tokenized segments across the given documents."""
        self.validate_data()
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "Word2Vec training requires gensim. Install with: pip install 'locisimiles[word2vec]'"
            ) from exc

        sentences = self._load_sentences(documents)
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
