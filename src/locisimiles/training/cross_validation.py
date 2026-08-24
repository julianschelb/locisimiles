# training/cross_validation.py
"""K-fold cross-validation utilities for reproducing the paper's evaluation protocol.

The paper reports mean±std across folds rather than a single train/test
split. These utilities split a :class:`~locisimiles.ground_truth.GroundTruth`
into folds grouped by ``query_id`` (so a query's positives and negatives
never straddle a train/eval boundary — matching the paper's query-stratified
K-fold protocol), then orchestrate train-on-(k-1)-folds /
evaluate-on-held-out-fold across all folds and aggregate the resulting
metrics.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Tuple

import pandas as pd

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth
from locisimiles.training.data import TrainingData

if TYPE_CHECKING:
    # Import-time only: locisimiles.pipeline.judge.lexical_classifier depends
    # on locisimiles.training.lexical, so a top-level import here would create
    # a circular import at runtime. Pipeline is only ever used as a type hint.
    from locisimiles.pipeline import Pipeline


def split_ground_truth_by_query(
    ground_truth: GroundTruth,
    n_folds: int,
    *,
    seed: int = 42,
) -> List[GroundTruth]:
    """Split a ``GroundTruth`` into ``n_folds`` folds grouped by ``query_id``.

    Every entry for a given query id lands in the same fold. Query ids are
    shuffled with ``seed`` before being assigned round-robin to folds, so
    fold sizes are as balanced as the query distribution allows.

    Args:
        ground_truth: Ground truth to split.
        n_folds: Number of folds (at least 2).
        seed: RNG seed for shuffling query ids before assignment.

    Returns:
        A list of ``n_folds`` disjoint ``GroundTruth`` objects whose union
        recovers all of ``ground_truth``'s entries.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    query_ids = sorted(ground_truth.query_ids(), key=str)
    rng = random.Random(seed)
    rng.shuffle(query_ids)

    fold_of_query = {query_id: i % n_folds for i, query_id in enumerate(query_ids)}
    fold_entries: List[list] = [[] for _ in range(n_folds)]
    for entry in ground_truth:
        fold_entries[fold_of_query[entry.query_id]].append(entry)

    return [GroundTruth(entries) for entries in fold_entries]


@dataclass
class CVFold:
    """One train/eval split of a K-fold cross-validation run.

    Attributes:
        index: Zero-based fold index.
        train_data: Training data — the union of every fold except this one.
        eval_data: This fold's held-out evaluation data.
    """

    index: int
    train_data: TrainingData
    eval_data: TrainingData


def make_cv_folds(
    *,
    query_doc: Document,
    source_doc: Document,
    ground_truth: GroundTruth,
    n_folds: int,
    seed: int = 42,
) -> List[CVFold]:
    """Build ``n_folds`` train/eval ``TrainingData`` splits, grouped by ``query_id``.

    Fold ``i``'s ``eval_data`` is the held-out fold produced by
    :func:`split_ground_truth_by_query`; its ``train_data`` is the union of
    every other fold.

    Args:
        query_doc: Query corpus.
        source_doc: Source corpus.
        ground_truth: Ground truth to split.
        n_folds: Number of folds (at least 2).
        seed: RNG seed forwarded to :func:`split_ground_truth_by_query`.

    Returns:
        A list of ``n_folds`` :class:`CVFold` objects.
    """
    fold_ground_truths = split_ground_truth_by_query(ground_truth, n_folds, seed=seed)

    folds: List[CVFold] = []
    for i, eval_gt in enumerate(fold_ground_truths):
        train_gt = GroundTruth()
        for j, gt in enumerate(fold_ground_truths):
            if j != i:
                train_gt = train_gt + gt
        folds.append(
            CVFold(
                index=i,
                train_data=TrainingData(query_doc, source_doc, train_gt),
                eval_data=TrainingData(query_doc, source_doc, eval_gt),
            )
        )
    return folds


def _aggregate(
    fold_metrics: Sequence[Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not fold_metrics:
        return {}, {}
    keys = fold_metrics[0].keys()
    mean = {key: statistics.mean(row[key] for row in fold_metrics) for key in keys}
    std = {
        key: (
            statistics.stdev([row[key] for row in fold_metrics]) if len(fold_metrics) > 1 else 0.0
        )
        for key in keys
    }
    return mean, std


@dataclass
class CVResult:
    """Aggregated results of a :func:`cross_validate` run.

    Attributes:
        fold_metrics: One metrics dict per fold, in fold order.
        mean: Per-metric mean across folds.
        std: Per-metric standard deviation across folds (``0.0`` for a
            single fold).
    """

    fold_metrics: List[Dict[str, float]]
    mean: Dict[str, float] = field(init=False)
    std: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.mean, self.std = _aggregate(self.fold_metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-fold metrics as a DataFrame, with ``mean``/``std`` summary rows appended."""
        df = pd.DataFrame(self.fold_metrics)
        df.insert(0, "fold", range(len(self.fold_metrics)))
        summary = pd.DataFrame([{"fold": "mean", **self.mean}, {"fold": "std", **self.std}])
        return pd.concat([df, summary], ignore_index=True)


def cross_validate(
    *,
    query_doc: Document,
    source_doc: Document,
    ground_truth: GroundTruth,
    n_folds: int,
    train_fn: Callable[[TrainingData], Any],
    evaluate_fn: Callable[[Any, TrainingData], Dict[str, float]],
    seed: int = 42,
) -> CVResult:
    """Run K-fold cross-validation, reproducing the paper's mean±std-across-folds protocol.

    For each fold, ``train_fn`` is called with the fold's training
    ``TrainingData`` (the union of every other fold) and must return a
    trained model/pipeline; ``evaluate_fn`` is then called with that return
    value and the fold's held-out ``TrainingData``, and must return a flat
    metric-name-to-value dict. Metrics are aggregated (mean/std) across
    folds. Folds are grouped by ``query_id`` (see
    :func:`split_ground_truth_by_query`), so a query's positives and
    negatives never straddle a train/eval boundary.

    Args:
        query_doc: Query corpus.
        source_doc: Source corpus.
        ground_truth: Full ground truth to cross-validate over.
        n_folds: Number of folds (at least 2).
        train_fn: Trains and returns a model/pipeline from one fold's training data.
        evaluate_fn: Evaluates a trained model/pipeline on one fold's held-out data.
        seed: RNG seed for the fold split.

    Returns:
        A :class:`CVResult` with per-fold metrics and their mean/std across folds.

    Example:
        ```python
        from locisimiles.training.cross_validation import cross_validate, evaluate_with_pipeline
        from locisimiles.training.classification import ClassificationTrainer, ClassificationTrainerConfig
        from locisimiles.pipeline import Pipeline
        from locisimiles.pipeline.generator import ExhaustiveCandidateGenerator
        from locisimiles.pipeline.judge import ClassificationJudge

        def train_fn(data):
            trainer = ClassificationTrainer(
                ClassificationTrainerConfig(output_dir="models/cv", epochs=4)
            )
            trainer.fit(data=data)
            return trainer.save()

        def evaluate_fn(model_path, eval_data):
            judge = ClassificationJudge(classification_name=str(model_path))
            pipeline = Pipeline(generator=ExhaustiveCandidateGenerator(), judge=judge)
            return evaluate_with_pipeline(pipeline, eval_data)

        result = cross_validate(
            query_doc=query_doc,
            source_doc=source_doc,
            ground_truth=ground_truth,
            n_folds=5,
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
        )
        print(result.mean, result.std)
        ```
    """
    folds = make_cv_folds(
        query_doc=query_doc,
        source_doc=source_doc,
        ground_truth=ground_truth,
        n_folds=n_folds,
        seed=seed,
    )
    fold_metrics: List[Dict[str, float]] = []
    for fold in folds:
        model = train_fn(fold.train_data)
        metrics = evaluate_fn(model, fold.eval_data)
        fold_metrics.append(dict(metrics))
    return CVResult(fold_metrics=fold_metrics)


def evaluate_with_pipeline(
    pipeline: Pipeline,
    eval_data: TrainingData,
    *,
    top_k: int = 10,
    average: str = "macro",
    **evaluator_kwargs: Any,
) -> Dict[str, float]:
    """Evaluate a pipeline on one fold's held-out data via ``IntertextEvaluator``.

    A convenience wrapper for the common ``evaluate_fn`` case in
    :func:`cross_validate`: builds an ``IntertextEvaluator`` from
    ``eval_data`` and returns ``evaluator.evaluate(average=average)`` as a
    flat dict.

    Args:
        pipeline: A trained/configured pipeline to evaluate.
        eval_data: Held-out ``TrainingData`` for one fold.
        top_k: Candidates per query segment, forwarded to ``IntertextEvaluator``.
        average: ``"macro"`` or ``"micro"``, forwarded to ``evaluator.evaluate()``.
        **evaluator_kwargs: Additional keyword arguments forwarded to
            ``IntertextEvaluator`` (e.g. ``threshold``).

    Returns:
        A flat metric-name-to-value dict for this fold.
    """
    from locisimiles.evaluator import IntertextEvaluator

    evaluator = IntertextEvaluator(
        query_doc=eval_data.query_doc,
        source_doc=eval_data.source_doc,
        ground_truth=eval_data.ground_truth,
        pipeline=pipeline,
        top_k=top_k,
        **evaluator_kwargs,
    )
    # IntertextEvaluator.evaluate() is annotated as Dict[str, float] but
    # actually returns a one-row DataFrame (a known, documented typing gap
    # in evaluator.py — see its mypy override).
    return evaluator.evaluate(average=average).iloc[0].to_dict()  # type: ignore[attr-defined]
