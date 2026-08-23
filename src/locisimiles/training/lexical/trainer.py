# training/lexical/trainer.py
"""Trainer for the lexical (LogReg/GBDT) classifier baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from locisimiles.training.artifacts import resolve_model_output_path
from locisimiles.training.base import BaseTrainer, TrainerConfig
from locisimiles.training.lexical.features import build_feature_matrix, fit_vectorizers


@dataclass(frozen=True)
class LexicalClassifierTrainerConfig(TrainerConfig):
    """Configuration for the lexical classifier trainer.

    ``train_path`` must point to a CSV with ``query_text``, ``corpus_text``,
    and ``label`` columns (``label`` may be an integer class id or a string
    class name, e.g. ``no_match`` / ``cit`` / ``cf`` for the three-class
    setup, or any two-class scheme for binary training).
    """

    classifier: Literal["logreg", "gbdt"] = "logreg"
    lemmatize: bool = True
    output_filename: str = "lexical_classifier.joblib"

    # LogisticRegression hyperparameters
    logreg_C: float = 1.0
    logreg_max_iter: int = 1000

    # HistGradientBoostingClassifier hyperparameters
    gbdt_max_iter: int = 300
    gbdt_learning_rate: float = 0.05
    gbdt_max_depth: int | None = None

    class_weight: str | dict | None = None
    label_names: dict[int, str] | None = field(default=None)


class LexicalClassifierTrainer(BaseTrainer):
    """Train a TF-IDF/Jaccard/overlap feature-based LogReg or GBDT classifier."""

    def __init__(self, config: LexicalClassifierTrainerConfig):
        super().__init__(config)
        self.vectorizers: dict[str, Any] | None = None
        self.model: Any | None = None
        self._label_to_id: dict[str, int] | None = None

    @property
    def cfg(self) -> LexicalClassifierTrainerConfig:
        return self.config  # type: ignore[return-value]

    def _load_rows(self) -> tuple[list[str], list[str], list[str]]:
        query_texts: list[str] = []
        corpus_texts: list[str] = []
        labels: list[str] = []
        with self.cfg.train_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"query_text", "corpus_text", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Training CSV is missing required columns: {sorted(missing)}")
            for row in reader:
                query_texts.append(row["query_text"])
                corpus_texts.append(row["corpus_text"])
                labels.append(str(row["label"]))
        if not query_texts:
            raise ValueError("No training rows found")
        return query_texts, corpus_texts, labels

    def _build_classifier(self) -> Any:
        if self.cfg.classifier == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=self.cfg.logreg_C,
                max_iter=self.cfg.logreg_max_iter,
                class_weight=self.cfg.class_weight,
                random_state=self.cfg.seed,
            )
        if self.cfg.classifier == "gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=self.cfg.gbdt_max_iter,
                learning_rate=self.cfg.gbdt_learning_rate,
                max_depth=self.cfg.gbdt_max_depth,
                class_weight=self.cfg.class_weight,
                random_state=self.cfg.seed,
            )
        raise ValueError(f"Unknown classifier: {self.cfg.classifier!r}")

    def fit(self, **kwargs: Any) -> Any:
        """Fit TF-IDF vectorizers and the configured classifier on paired training data."""
        self.validate_data()
        query_texts, corpus_texts, labels = self._load_rows()

        sorted_labels = sorted(set(labels))
        self._label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        y = [self._label_to_id[label] for label in labels]

        pooled_texts = query_texts + corpus_texts
        self.vectorizers = fit_vectorizers(
            pooled_texts, lemmatize=self.cfg.lemmatize, lowercase=self.cfg.lowercase
        )
        X = build_feature_matrix(
            query_texts,
            corpus_texts,
            self.vectorizers,
            lemmatize=self.cfg.lemmatize,
            lowercase=self.cfg.lowercase,
        )

        self.model = self._build_classifier()
        self.model.fit(X, y, **kwargs)
        return self.model

    def save(self, **kwargs: Any) -> Path:
        """Persist vectorizers, classifier, and label mapping to one joblib artifact."""
        if self.model is None or self.vectorizers is None or self._label_to_id is None:
            raise ValueError("No trained model available. Call fit() first.")
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Saving the lexical classifier requires joblib. "
                "Install it with: pip install 'locisimiles[lexical]'"
            ) from exc

        output_path = resolve_model_output_path(self.cfg.output_dir, self.cfg.output_filename)
        label_names = self.cfg.label_names or {
            idx: label for label, idx in self._label_to_id.items()
        }
        artifact = {
            "vectorizers": self.vectorizers,
            "model": self.model,
            "label_to_id": self._label_to_id,
            "label_names": label_names,
            "lemmatize": self.cfg.lemmatize,
            "lowercase": self.cfg.lowercase,
        }
        joblib.dump(artifact, output_path)
        return output_path

    def load_artifacts(self, path: str | Path) -> Any:
        """Load a previously saved lexical classifier artifact."""
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Loading the lexical classifier requires joblib. "
                "Install it with: pip install 'locisimiles[lexical]'"
            ) from exc

        artifact = joblib.load(path)
        self.vectorizers = artifact["vectorizers"]
        self.model = artifact["model"]
        self._label_to_id = artifact["label_to_id"]
        return artifact
