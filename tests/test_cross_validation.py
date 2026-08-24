"""Tests for K-fold cross-validation utilities."""

from __future__ import annotations

import pytest

from locisimiles.ground_truth import GroundTruth
from locisimiles.training.cross_validation import (
    CVResult,
    cross_validate,
    evaluate_with_pipeline,
    make_cv_folds,
    split_ground_truth_by_query,
)
from locisimiles.training.data import TrainingData


@pytest.fixture
def ten_query_ground_truth() -> GroundTruth:
    """10 queries, each with a positive and negative row against the same source id."""
    rows = []
    for i in range(10):
        rows.append({"query_id": f"q{i}", "source_id": f"s{i}", "label": "cit"})
        rows.append({"query_id": f"q{i}", "source_id": f"s{i}_neg", "label": "no_match"})
    return GroundTruth(rows)


class TestSplitGroundTruthByQuery:
    def test_union_recovers_all_entries(self, ten_query_ground_truth):
        folds = split_ground_truth_by_query(ten_query_ground_truth, 5, seed=1)
        total = sum(len(fold) for fold in folds)
        assert total == len(ten_query_ground_truth)

    def test_query_never_split_across_folds(self, ten_query_ground_truth):
        folds = split_ground_truth_by_query(ten_query_ground_truth, 5, seed=1)
        for fold in folds:
            query_ids_in_fold = {entry.query_id for entry in fold}
            for query_id in query_ids_in_fold:
                # every entry for this query id, across the WHOLE ground truth,
                # must appear only in this one fold.
                total_for_query = sum(1 for e in ten_query_ground_truth if e.query_id == query_id)
                in_fold = sum(1 for e in fold if e.query_id == query_id)
                assert in_fold == total_for_query

    def test_folds_are_disjoint(self, ten_query_ground_truth):
        folds = split_ground_truth_by_query(ten_query_ground_truth, 5, seed=1)
        seen_pairs = set()
        for fold in folds:
            for entry in fold:
                pair = (entry.query_id, entry.source_id)
                assert pair not in seen_pairs
                seen_pairs.add(pair)

    def test_reproducible_for_fixed_seed(self, ten_query_ground_truth):
        first = split_ground_truth_by_query(ten_query_ground_truth, 5, seed=7)
        second = split_ground_truth_by_query(ten_query_ground_truth, 5, seed=7)
        first_ids = [{e.query_id for e in fold} for fold in first]
        second_ids = [{e.query_id for e in fold} for fold in second]
        assert first_ids == second_ids

    def test_n_folds_less_than_two_raises(self, ten_query_ground_truth):
        with pytest.raises(ValueError, match="n_folds must be at least 2"):
            split_ground_truth_by_query(ten_query_ground_truth, 1)

    def test_number_of_folds_matches_request(self, ten_query_ground_truth):
        folds = split_ground_truth_by_query(ten_query_ground_truth, 5, seed=1)
        assert len(folds) == 5


class TestMakeCVFolds:
    def test_train_and_eval_are_complementary(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        folds = make_cv_folds(
            query_doc=query_document, source_doc=source_document, ground_truth=gt, n_folds=3
        )
        assert len(folds) == 3
        for fold in folds:
            eval_ids = {e.query_id for e in fold.eval_data.ground_truth}
            train_ids = {e.query_id for e in fold.train_data.ground_truth}
            assert eval_ids.isdisjoint(train_ids)
            assert len(fold.eval_data) + len(fold.train_data) == len(gt)

    def test_documents_passed_through(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        folds = make_cv_folds(
            query_doc=query_document, source_doc=source_document, ground_truth=gt, n_folds=2
        )
        for fold in folds:
            assert fold.train_data.query_doc is query_document
            assert fold.train_data.source_doc is source_document
            assert fold.eval_data.query_doc is query_document
            assert fold.eval_data.source_doc is source_document

    def test_fold_index_matches_position(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        folds = make_cv_folds(
            query_doc=query_document, source_doc=source_document, ground_truth=gt, n_folds=3
        )
        assert [fold.index for fold in folds] == [0, 1, 2]


class TestCrossValidate:
    def test_calls_train_and_evaluate_once_per_fold(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        train_calls = []
        eval_calls = []

        def train_fn(data: TrainingData):
            train_calls.append(len(data))
            return "trained-model"

        def evaluate_fn(model, eval_data: TrainingData):
            eval_calls.append((model, len(eval_data)))
            return {"f1": 0.5, "precision": 0.6}

        result = cross_validate(
            query_doc=query_document,
            source_doc=source_document,
            ground_truth=gt,
            n_folds=3,
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
        )
        assert len(train_calls) == 3
        assert len(eval_calls) == 3
        assert all(model == "trained-model" for model, _ in eval_calls)
        assert isinstance(result, CVResult)
        assert len(result.fold_metrics) == 3

    def test_aggregates_mean_and_std(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        fold_scores = iter([0.2, 0.4, 0.6])

        result = cross_validate(
            query_doc=query_document,
            source_doc=source_document,
            ground_truth=gt,
            n_folds=3,
            train_fn=lambda data: None,
            evaluate_fn=lambda model, eval_data: {"f1": next(fold_scores)},
        )
        assert result.mean["f1"] == pytest.approx(0.4)
        assert result.std["f1"] == pytest.approx(0.2)

    def test_single_fold_std_is_zero(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        result = cross_validate(
            query_doc=query_document,
            source_doc=source_document,
            ground_truth=gt,
            n_folds=2,
            train_fn=lambda data: None,
            evaluate_fn=lambda model, eval_data: {"f1": 0.5},
        )
        assert result.std["f1"] == 0.0

    def test_to_dataframe_shape(self, query_document, source_document):
        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": "cit"} for seg in query_document]
        )
        result = cross_validate(
            query_doc=query_document,
            source_doc=source_document,
            ground_truth=gt,
            n_folds=3,
            train_fn=lambda data: None,
            evaluate_fn=lambda model, eval_data: {"f1": 0.5},
        )
        df = result.to_dataframe()
        assert len(df) == 5  # 3 folds + mean + std rows
        assert list(df["fold"])[-2:] == ["mean", "std"]


class TestEvaluateWithPipeline:
    def test_returns_flat_metric_dict(self, query_document, source_document):
        from locisimiles.pipeline import Pipeline
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
        from locisimiles.pipeline.judge.identity import IdentityJudge

        gt = GroundTruth(
            [{"query_id": seg.id, "source_id": "s1", "label": 1} for seg in query_document]
        )
        eval_data = TrainingData(query_document, source_document, gt)
        pipeline = Pipeline(generator=ExhaustiveCandidateGenerator(), judge=IdentityJudge())

        metrics = evaluate_with_pipeline(pipeline, eval_data, top_k=5)

        assert isinstance(metrics, dict)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
