# training/classification/threshold.py
"""Threshold tuning and application for the classification trainer/judge.

A tuned ``threshold.json`` is useless if nothing applies it —
:class:`~locisimiles.pipeline.judge.classification.ClassificationJudge` only
supports a single uniform cutoff supplied by the caller, and doesn't know
about per-class one-vs-rest thresholds or a 3-class tie-break rule at all.
This module closes that loop as an explicit, standalone post-processing
step: a caller runs ``judge.judge(...)`` as normal, then optionally applies
:func:`apply_thresholds_to_judgments` to get the tuned-threshold decision —
without any change to ``ClassificationJudge`` itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from locisimiles.pipeline._types import CandidateJudge, CandidateJudgeOutput

# =============================================================================
# Data model
# =============================================================================


@dataclass
class ThresholdSet:
    """Per-class decision thresholds tuned by :meth:`ClassificationTrainer.tune_threshold`.

    Attributes:
        thresholds: Mapping from positive class label to its tuned
            one-vs-rest decision threshold.
        method: Threshold-tuning method used (``"max_f1"`` or ``"youden"``).
        tie_break: Rule used to resolve ties when multiple positive classes
            clear their threshold for the same pair. Only ``"max_probability"``
            (compare class probabilities) is currently supported.
    """

    thresholds: Dict[str, float]
    method: str = "max_f1"
    tie_break: str = "max_probability"

    def to_json(self, path: Union[str, Path]) -> Path:
        """Persist this threshold set as a JSON sidecar file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "thresholds": self.thresholds,
                    "method": self.method,
                    "tie_break": self.tie_break,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> ThresholdSet:
        """Load a threshold set previously written by :meth:`to_json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            thresholds=dict(data["thresholds"]),
            method=data.get("method", "max_f1"),
            tie_break=data.get("tie_break", "max_probability"),
        )


# =============================================================================
# Threshold tuning
# =============================================================================


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _best_threshold_for_class(
    probabilities: Sequence[Sequence[float]],
    gold_labels: Sequence[str],
    *,
    class_id: int,
    class_label: str,
    method: str,
) -> float:
    """Sweep thresholds 0.01-0.99 for one positive class (one-vs-rest)."""
    thresholds = [round(t * 0.01, 2) for t in range(1, 100)]
    class_probs = [row[class_id] for row in probabilities]
    is_positive_gold = [label == class_label for label in gold_labels]

    # score every candidate threshold and keep the best one
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        predicted = [p >= threshold for p in class_probs]
        tp = sum(1 for pred, gold in zip(predicted, is_positive_gold) if pred and gold)
        fp = sum(1 for pred, gold in zip(predicted, is_positive_gold) if pred and not gold)
        fn = sum(1 for pred, gold in zip(predicted, is_positive_gold) if not pred and gold)
        tn = sum(1 for pred, gold in zip(predicted, is_positive_gold) if not pred and not gold)

        if method == "max_f1":
            score = _f1(tp, fp, fn)
        elif method == "youden":
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
        else:
            raise ValueError(f"Unknown threshold-tuning method: {method!r}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def tune_threshold(
    *,
    probabilities: Sequence[Sequence[float]],
    gold_labels: Sequence[str],
    id_to_label: Dict[int, str],
    method: str = "max_f1",
    negative_label: str = "no_match",
) -> ThresholdSet:
    """Tune one-vs-rest decision thresholds for every positive class.

    Args:
        probabilities: One row of per-class probabilities per evaluation pair.
        gold_labels: Gold label string per evaluation pair.
        id_to_label: Mapping from class id (probability column index) to label.
        method: ``"max_f1"`` (default) or ``"youden"``.
        negative_label: Label treated as the non-link class (excluded from tuning).

    Returns:
        A :class:`ThresholdSet` with one tuned threshold per positive class.
    """
    if method not in {"max_f1", "youden"}:
        raise ValueError(f"Unknown threshold-tuning method: {method!r}")

    # tune one threshold per positive class, skipping the negative class
    positive_class_ids = [
        class_id for class_id, label in id_to_label.items() if label != negative_label
    ]
    thresholds: Dict[str, float] = {}
    for class_id in positive_class_ids:
        class_label = id_to_label[class_id]
        thresholds[class_label] = _best_threshold_for_class(
            probabilities,
            gold_labels,
            class_id=class_id,
            class_label=class_label,
            method=method,
        )
    return ThresholdSet(thresholds=thresholds, method=method)


# =============================================================================
# Threshold application
# =============================================================================


def apply_thresholds(
    class_probabilities: Dict[str, float],
    thresholds: ThresholdSet,
    *,
    negative_label: str = "no_match",
) -> Tuple[str, Optional[int]]:
    """Apply tuned one-vs-rest thresholds and the tie-break rule to one probability row.

    Args:
        class_probabilities: Mapping from class label to probability, e.g.
            as emitted by ``ClassificationJudge`` with ``emit_class_metadata=True``.
        thresholds: Tuned thresholds from :func:`tune_threshold`.
        negative_label: Label returned when no positive class clears its threshold.

    Returns:
        ``(predicted_label, predicted_class_id)``. ``predicted_class_id`` is
        always ``None`` here, since a ``ThresholdSet`` only carries labels,
        not a label-to-id mapping.
    """
    cleared = [
        (label, class_probabilities.get(label, 0.0))
        for label, threshold in thresholds.thresholds.items()
        if class_probabilities.get(label, 0.0) >= threshold
    ]
    if not cleared:
        return negative_label, None
    # Tie-break: compare probabilities (the only supported rule today).
    best_label, _ = max(cleared, key=lambda item: item[1])
    return best_label, None


def apply_thresholds_to_judgments(
    judgments: CandidateJudgeOutput,
    thresholds: ThresholdSet,
    *,
    negative_label: str = "no_match",
) -> CandidateJudgeOutput:
    """Re-decide ``judgment_score``/``predicted_label`` using tuned thresholds.

    Runs :func:`apply_thresholds` over every judgment's existing
    ``class_probabilities`` and returns a **new** ``CandidateJudgeOutput``
    with ``judgment_score``/``predicted_label``/``predicted_class_id``
    overwritten accordingly. Judgments without ``class_probabilities``
    (e.g. from a judge run without ``emit_class_metadata=True``) are passed
    through unchanged.

    Args:
        judgments: Output of ``ClassificationJudge.judge()``.
        thresholds: Tuned thresholds from :func:`tune_threshold`/`ThresholdSet.from_json`.
        negative_label: Label treated as the non-link class.

    Returns:
        A new ``CandidateJudgeOutput`` with thresholded decisions.
    """
    result: CandidateJudgeOutput = {}
    for query_id, items in judgments.items():
        new_items: List[CandidateJudge] = []
        for item in items:
            # nothing to re-decide without class probabilities
            if not item.class_probabilities:
                new_items.append(item)
                continue
            predicted_label, predicted_class_id = apply_thresholds(
                item.class_probabilities, thresholds, negative_label=negative_label
            )
            new_items.append(
                CandidateJudge(
                    segment=item.segment,
                    candidate_score=item.candidate_score,
                    judgment_score=item.class_probabilities.get(predicted_label, 0.0),
                    predicted_class_id=predicted_class_id,
                    predicted_label=predicted_label,
                    class_probabilities=item.class_probabilities,
                )
            )
        result[query_id] = new_items
    return result
