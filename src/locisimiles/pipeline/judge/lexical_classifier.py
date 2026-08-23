# pipeline/judge/lexical_classifier.py
"""Lexical (LogReg/GBDT) classifier judge — the benchmark's non-neural baseline."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from locisimiles.document import Document
from locisimiles.pipeline._types import (
    CandidateGeneratorOutput,
    CandidateJudge,
    CandidateJudgeOutput,
)
from locisimiles.pipeline.judge._base import CandidateJudgeBase
from locisimiles.pipeline.judge._positive_classes import (
    DEFAULT_NEGATIVE_LABELS,
    normalise_label,
    prediction_from_probabilities,
)
from locisimiles.training.lexical.features import build_feature_matrix


class LexicalClassifierJudge(CandidateJudgeBase):
    """Judge candidates using a trained TF-IDF/Jaccard/overlap LogReg or GBDT classifier.

    Loads an artifact saved by :class:`~locisimiles.training.lexical.LexicalClassifierTrainer`
    (fitted vectorizers + a scikit-learn classifier) and scores each
    query/candidate pair with the same feature pipeline used at training
    time. Works for both the benchmark's two-class (match / no-match) and
    three-class (``no_match`` / ``cit`` / ``cf``) lexical classifiers — the
    number of classes is inferred from the loaded artifact, following the
    same binary/multiclass rules as :class:`~locisimiles.pipeline.judge.classification.ClassificationJudge`.

    Args:
        artifact_path: Path to a ``.joblib`` artifact produced by
            ``LexicalClassifierTrainer.save()``.
        pos_class_idx: Index of the positive class. Kept for binary models
            and as a fallback when no positive classes can be inferred.
        positive_class_ids: Optional class ids whose probabilities are summed
            into ``judgment_score``.
        positive_labels: Optional class labels whose probabilities are summed
            into ``judgment_score``.
        negative_labels: Optional class labels treated as non-links. When no
            positive classes are provided, all non-negative classes are
            treated as positive for multiclass models.
        emit_class_metadata: Whether to attach predicted labels and class
            probabilities to output results. Defaults to automatic behavior:
            enabled for multiclass or explicitly label-configured models,
            disabled for default binary models.
    """

    def __init__(
        self,
        *,
        artifact_path: str,
        pos_class_idx: int = 1,
        positive_class_ids: Sequence[int] | None = None,
        positive_labels: Sequence[str] | None = None,
        negative_labels: Sequence[str] | None = None,
        emit_class_metadata: bool | None = None,
    ):
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "LexicalClassifierJudge requires joblib. "
                "Install it with: pip install 'locisimiles[lexical]'"
            ) from exc

        artifact = joblib.load(artifact_path)
        self.vectorizers = artifact["vectorizers"]
        self.model = artifact["model"]
        self.label_names: Dict[int, str] = {
            int(idx): str(label) for idx, label in artifact["label_names"].items()
        }
        self.lemmatize = bool(artifact.get("lemmatize", True))
        self.lowercase = bool(artifact.get("lowercase", True))

        self.pos_class_idx = pos_class_idx
        self.positive_class_ids = (
            list(positive_class_ids) if positive_class_ids is not None else None
        )
        self.positive_labels = (
            {normalise_label(label) for label in positive_labels}
            if positive_labels is not None
            else None
        )
        self.negative_labels = DEFAULT_NEGATIVE_LABELS | {
            normalise_label(label) for label in (negative_labels or [])
        }

        num_labels = len(self.label_names)
        has_explicit_class_config = any(
            value is not None for value in (positive_class_ids, positive_labels, negative_labels)
        )
        self.emit_class_metadata = (
            num_labels > 2 or has_explicit_class_config
            if emit_class_metadata is None
            else emit_class_metadata
        )

    def _label_for_class_id(self, class_id: int) -> str:
        return self.label_names.get(class_id, f"LABEL_{class_id}")

    def judge(
        self,
        *,
        query: Document,
        candidates: CandidateGeneratorOutput,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Score each (query, candidate) pair with the trained lexical classifier."""
        query_by_id = {str(segment.id): segment for segment in query.segments.values()}

        flat_query_texts: List[str] = []
        flat_corpus_texts: List[str] = []
        flat_index: List[tuple[str, int]] = []
        for query_id, candidate_list in candidates.items():
            query_segment = query_by_id.get(str(query_id))
            if query_segment is None:
                continue
            for position, candidate in enumerate(candidate_list):
                flat_query_texts.append(query_segment.text)
                flat_corpus_texts.append(candidate.segment.text)
                flat_index.append((str(query_id), position))

        results: CandidateJudgeOutput = {str(qid): [] for qid in candidates}
        if not flat_query_texts:
            return results

        X = build_feature_matrix(
            flat_query_texts,
            flat_corpus_texts,
            self.vectorizers,
            lemmatize=self.lemmatize,
            lowercase=self.lowercase,
        )
        probabilities = self.model.predict_proba(X)

        for row, (query_id, position) in enumerate(flat_index):
            candidate = candidates[query_id][position]
            prediction = prediction_from_probabilities(
                probabilities[row],
                label_for_class_id=self._label_for_class_id,
                positive_class_ids=self.positive_class_ids,
                positive_labels=self.positive_labels,
                negative_labels=self.negative_labels,
                pos_class_idx=self.pos_class_idx,
            )
            results[query_id].append(
                CandidateJudge(
                    segment=candidate.segment,
                    candidate_score=candidate.score,
                    judgment_score=prediction.judgment_score,
                    predicted_class_id=(
                        prediction.predicted_class_id if self.emit_class_metadata else None
                    ),
                    predicted_label=(
                        prediction.predicted_label if self.emit_class_metadata else None
                    ),
                    class_probabilities=(
                        prediction.class_probabilities if self.emit_class_metadata else None
                    ),
                )
            )

        return results
