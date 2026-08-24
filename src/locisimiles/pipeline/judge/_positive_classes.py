# pipeline/judge/_positive_classes.py
"""Shared binary/multiclass positive-class resolution for classifier judges.

Used by both :class:`~locisimiles.pipeline.judge.classification.ClassificationJudge`
(transformer models) and :class:`~locisimiles.pipeline.judge.lexical_classifier.LexicalClassifierJudge`
(sklearn models) so the two follow identical rules for turning a class
probability vector into a ``judgment_score``, whether the underlying model is
binary or multiclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

# =============================================================================
# Label normalisation
# =============================================================================

DEFAULT_NEGATIVE_LABELS = {
    "0",
    "label_0",
    "negative",
    "no",
    "none",
    "no_match",
    "no-match",
    "no match",
    "not_intertext",
    "non_link",
    "non-link",
}


def normalise_label(label: object) -> str:
    """Canonicalise class labels for matching user/model metadata."""
    return str(label).strip().lower().replace("-", "_").replace(" ", "_").rstrip(".")


# =============================================================================
# Positive-class resolution
# =============================================================================


@dataclass(frozen=True)
class ClassifierPrediction:
    """One row of resolved classifier output."""

    judgment_score: float
    predicted_class_id: int
    predicted_label: str
    class_probabilities: Dict[str, float]


def resolve_positive_class_ids(
    *,
    num_labels: int,
    label_for_class_id: Callable[[int], str],
    positive_class_ids: Sequence[int] | None,
    positive_labels: set[str] | None,
    negative_labels: set[str],
    pos_class_idx: int,
) -> List[int]:
    """Resolve which class ids count as positive intertextual links."""
    if positive_class_ids is not None:
        return [idx for idx in positive_class_ids if 0 <= idx < num_labels]

    if positive_labels is not None:
        return [
            idx
            for idx in range(num_labels)
            if normalise_label(label_for_class_id(idx)) in positive_labels
        ]

    if num_labels <= 2:
        return [pos_class_idx] if 0 <= pos_class_idx < num_labels else []

    return [
        idx
        for idx in range(num_labels)
        if normalise_label(label_for_class_id(idx)) not in negative_labels
    ]


def prediction_from_probabilities(
    probabilities: Sequence[float],
    *,
    label_for_class_id: Callable[[int], str],
    positive_class_ids: Sequence[int] | None,
    positive_labels: set[str] | None,
    negative_labels: set[str],
    pos_class_idx: int,
) -> ClassifierPrediction:
    """Build classifier metadata from one probability row (binary or multiclass)."""
    prob_list = [float(probability) for probability in probabilities]
    predicted_class_id = max(range(len(prob_list)), key=prob_list.__getitem__)
    positive_ids = resolve_positive_class_ids(
        num_labels=len(prob_list),
        label_for_class_id=label_for_class_id,
        positive_class_ids=positive_class_ids,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        pos_class_idx=pos_class_idx,
    )
    judgment_score = sum(prob_list[idx] for idx in positive_ids)
    class_probabilities = {
        label_for_class_id(idx): probability for idx, probability in enumerate(prob_list)
    }
    return ClassifierPrediction(
        judgment_score=judgment_score,
        predicted_class_id=predicted_class_id,
        predicted_label=label_for_class_id(predicted_class_id),
        class_probabilities=class_probabilities,
    )
