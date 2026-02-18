"""
Unit tests for locisimiles.evaluator module.
Tests metric helper functions and IntertextEvaluator class.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from locisimiles.document import Document, TextSegment
from locisimiles.evaluator import (
    IntertextEvaluator,
    _f1,
    _fn_rate,
    _fp_rate,
    _precision,
    _recall,
    _smr,
)
from locisimiles.pipeline._types import CandidateJudge

# =================== METRIC HELPER TESTS ===================


class TestPrecision:
    """Tests for _precision function."""

    def test_precision_basic(self):
        """Test basic precision calculation."""
        # precision = tp / (tp + fp) = 5 / 7 ≈ 0.714
        result = _precision(tp=5, fp=2)
        assert pytest.approx(result, rel=1e-3) == 0.714

    def test_precision_perfect(self):
        """Test precision = 1.0 when no false positives."""
        result = _precision(tp=10, fp=0)
        assert result == 1.0

    def test_precision_zero_tp(self):
        """Test precision = 0 when no true positives."""
        result = _precision(tp=0, fp=5)
        assert result == 0.0

    def test_precision_zero_both(self):
        """Test precision = 0 when both tp and fp are zero."""
        result = _precision(tp=0, fp=0)
        assert result == 0.0


class TestRecall:
    """Tests for _recall function."""

    def test_recall_basic(self):
        """Test basic recall calculation."""
        # recall = tp / (tp + fn) = 5 / 8 = 0.625
        result = _recall(tp=5, fn=3)
        assert pytest.approx(result, rel=1e-3) == 0.625

    def test_recall_perfect(self):
        """Test recall = 1.0 when no false negatives."""
        result = _recall(tp=10, fn=0)
        assert result == 1.0

    def test_recall_zero_tp(self):
        """Test recall = 0 when no true positives."""
        result = _recall(tp=0, fn=5)
        assert result == 0.0

    def test_recall_zero_both(self):
        """Test recall = 0 when both tp and fn are zero."""
        result = _recall(tp=0, fn=0)
        assert result == 0.0


class TestF1Score:
    """Tests for _f1 function."""

    def test_f1_basic(self):
        """Test basic F1 score calculation."""
        # f1 = 2 * p * r / (p + r) = 2 * 0.8 * 0.6 / 1.4 ≈ 0.686
        result = _f1(p=0.8, r=0.6)
        assert pytest.approx(result, rel=1e-3) == 0.686

    def test_f1_perfect(self):
        """Test F1 = 1.0 when both precision and recall are 1.0."""
        result = _f1(p=1.0, r=1.0)
        assert result == 1.0

    def test_f1_zero_precision(self):
        """Test F1 = 0 when precision is 0."""
        result = _f1(p=0.0, r=0.8)
        assert result == 0.0

    def test_f1_zero_recall(self):
        """Test F1 = 0 when recall is 0."""
        result = _f1(p=0.8, r=0.0)
        assert result == 0.0

    def test_f1_zero_both(self):
        """Test F1 = 0 when both are zero."""
        result = _f1(p=0.0, r=0.0)
        assert result == 0.0

    def test_f1_equal_precision_recall(self):
        """Test F1 equals P and R when they are equal."""
        result = _f1(p=0.7, r=0.7)
        assert pytest.approx(result, rel=1e-3) == 0.7


class TestSMR:
    """Tests for _smr (Source Match Rate) function."""

    def test_smr_basic(self):
        """Test SMR calculation: (fp + fn) / total."""
        # total = 5 + 2 + 3 + 10 = 20
        # smr = (2 + 3) / 20 = 0.25
        result = _smr(tp=5, fp=2, fn=3, tn=10)
        assert pytest.approx(result, rel=1e-3) == 0.25

    def test_smr_perfect(self):
        """Test SMR = 0 when no errors."""
        result = _smr(tp=10, fp=0, fn=0, tn=90)
        assert result == 0.0

    def test_smr_all_errors(self):
        """Test SMR = 1.0 when all predictions are wrong."""
        result = _smr(tp=0, fp=5, fn=5, tn=0)
        assert result == 1.0


class TestFPRate:
    """Tests for _fp_rate function."""

    def test_fp_rate_basic(self):
        """Test FP rate calculation: fp / total."""
        # total = 5 + 3 + 2 + 10 = 20
        # fp_rate = 3 / 20 = 0.15
        result = _fp_rate(tp=5, fp=3, fn=2, tn=10)
        assert pytest.approx(result, rel=1e-3) == 0.15

    def test_fp_rate_zero(self):
        """Test FP rate = 0 when no false positives."""
        result = _fp_rate(tp=10, fp=0, fn=5, tn=85)
        assert result == 0.0


class TestFNRate:
    """Tests for _fn_rate function."""

    def test_fn_rate_basic(self):
        """Test FN rate calculation: fn / total."""
        # total = 5 + 3 + 4 + 8 = 20
        # fn_rate = 4 / 20 = 0.2
        result = _fn_rate(tp=5, fp=3, fn=4, tn=8)
        assert pytest.approx(result, rel=1e-3) == 0.2

    def test_fn_rate_zero(self):
        """Test FN rate = 0 when no false negatives."""
        result = _fn_rate(tp=10, fp=5, fn=0, tn=85)
        assert result == 0.0


# =================== INTERTEXT EVALUATOR TESTS ===================


class TestIntertextEvaluatorLoadGoldLabels:
    """Tests for IntertextEvaluator ground truth loading."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline that returns empty predictions."""
        mock = MagicMock()
        mock.run.return_value = {
            "q1": [],
            "q2": [],
            "q3": [],
        }
        return mock

    @pytest.fixture
    def evaluator_docs(self, temp_dir):
        """Create query and source documents for testing."""
        query_csv = temp_dir / "query.csv"
        query_csv.write_text(
            "seg_id,text\nq1,Query one text.\nq2,Query two text.\nq3,Query three text.\n",
            encoding="utf-8",
        )
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,Source one text.\ns2,Source two text.\ns3,Source three text.\n",
            encoding="utf-8",
        )
        return Document(query_csv), Document(source_csv)

    def test_load_gold_labels_from_csv(self, evaluator_docs, ground_truth_csv, mock_pipeline):
        """Test loading gold labels from CSV path."""
        query_doc, source_doc = evaluator_docs
        evaluator = IntertextEvaluator(
            query_doc=query_doc,
            source_doc=source_doc,
            ground_truth_csv=str(ground_truth_csv),
            pipeline=mock_pipeline,
            threshold=0.5,
        )
        # Check that gold labels were loaded
        assert ("q1", "s1") in evaluator.gold_labels
        assert evaluator.gold_labels[("q1", "s1")] == 1
        assert evaluator.gold_labels[("q1", "s2")] == 0
        assert evaluator.gold_labels[("q2", "s2")] == 1

    def test_load_gold_labels_from_dataframe(self, evaluator_docs, mock_pipeline):
        """Test loading gold labels from DataFrame."""
        query_doc, source_doc = evaluator_docs
        gt_df = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2"],
                "source_id": ["s1", "s2", "s1"],
                "label": [1, 0, 1],
            }
        )
        evaluator = IntertextEvaluator(
            query_doc=query_doc,
            source_doc=source_doc,
            ground_truth_csv=gt_df,
            pipeline=mock_pipeline,
            threshold=0.5,
        )
        assert evaluator.gold_labels[("q1", "s1")] == 1
        assert evaluator.gold_labels[("q2", "s1")] == 1


class TestIntertextEvaluatorMetrics:
    """Tests for IntertextEvaluator metric computation."""

    @pytest.fixture
    def configured_evaluator(self, temp_dir):
        """Create a fully configured evaluator with known predictions."""
        # Create documents
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query one.\nq2,Query two.\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,Source one.\ns2,Source two.\ns3,Source three.\n", encoding="utf-8"
        )
        query_doc = Document(query_csv)
        source_doc = Document(source_csv)

        # Create ground truth
        gt_csv = temp_dir / "gt.csv"
        gt_csv.write_text(
            "query_id,source_id,label\nq1,s1,1\nq1,s2,0\nq1,s3,0\nq2,s1,0\nq2,s2,1\nq2,s3,1\n",
            encoding="utf-8",
        )

        # Create mock pipeline with specific predictions
        mock_pipeline = MagicMock()
        s1 = TextSegment("Source one.", "s1", row_id=0)
        s2 = TextSegment("Source two.", "s2", row_id=1)
        s3 = TextSegment("Source three.", "s3", row_id=2)

        # Predictions: prob > 0.5 means positive
        mock_pipeline.run.return_value = {
            "q1": [
                CandidateJudge(segment=s1, candidate_score=0.9, judgment_score=0.8),
                CandidateJudge(segment=s2, candidate_score=0.7, judgment_score=0.3),
                CandidateJudge(segment=s3, candidate_score=0.5, judgment_score=0.6),
            ],  # Pred: s1=1, s2=0, s3=1
            "q2": [
                CandidateJudge(segment=s1, candidate_score=0.8, judgment_score=0.7),
                CandidateJudge(segment=s2, candidate_score=0.6, judgment_score=0.9),
                CandidateJudge(segment=s3, candidate_score=0.4, judgment_score=0.2),
            ],  # Pred: s1=1, s2=1, s3=0
        }

        evaluator = IntertextEvaluator(
            query_doc=query_doc,
            source_doc=source_doc,
            ground_truth_csv=str(gt_csv),
            pipeline=mock_pipeline,
            threshold=0.5,
        )
        return evaluator

    def test_evaluate_single_query(self, configured_evaluator):
        """Test evaluating a single query."""
        result = configured_evaluator.evaluate_single_query("q1")
        assert "query_id" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "tp" in result
        assert "fp" in result
        assert "fn" in result
        assert "tn" in result

    def test_evaluate_all_queries(self, configured_evaluator):
        """Test evaluating all queries returns DataFrame."""
        df = configured_evaluator.evaluate_all_queries()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two queries
        assert "precision" in df.columns
        assert "recall" in df.columns

    def test_confusion_matrix(self, configured_evaluator):
        """Test confusion matrix shape and values."""
        cm = configured_evaluator.confusion_matrix("q1")
        assert cm.shape == (2, 2)
        assert isinstance(cm, np.ndarray)
        # Should be [[TP, FP], [FN, TN]]
        assert cm.sum() == 3  # 3 source segments

    def test_evaluate_macro_average(self, configured_evaluator):
        """Test macro averaging computes mean of per-query metrics."""
        result = configured_evaluator.evaluate(average="macro")
        assert isinstance(result, pd.DataFrame)
        assert "precision" in result.columns

    def test_evaluate_micro_average(self, configured_evaluator):
        """Test micro averaging pools counts before computing metrics."""
        result = configured_evaluator.evaluate(average="micro")
        assert isinstance(result, pd.DataFrame)
        assert "precision" in result.columns

    def test_evaluate_invalid_average(self, configured_evaluator):
        """Test invalid average parameter raises error."""
        with pytest.raises(ValueError, match="average must be"):
            configured_evaluator.evaluate(average="invalid")


class TestIntertextEvaluatorThreshold:
    """Tests for threshold optimization in IntertextEvaluator."""

    @pytest.fixture
    def evaluator_for_threshold(self, temp_dir):
        """Create evaluator for threshold testing."""
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query.\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source one.\ns2,Source two.\n", encoding="utf-8")
        gt_csv = temp_dir / "gt.csv"
        gt_csv.write_text("query_id,source_id,label\nq1,s1,1\nq1,s2,0\n", encoding="utf-8")

        mock_pipeline = MagicMock()
        s1 = TextSegment("Source one.", "s1", row_id=0)
        s2 = TextSegment("Source two.", "s2", row_id=1)
        mock_pipeline.run.return_value = {
            "q1": [
                CandidateJudge(segment=s1, candidate_score=0.9, judgment_score=0.75),
                CandidateJudge(segment=s2, candidate_score=0.5, judgment_score=0.25),
            ],
        }

        return IntertextEvaluator(
            query_doc=Document(query_csv),
            source_doc=Document(source_csv),
            ground_truth_csv=str(gt_csv),
            pipeline=mock_pipeline,
            threshold=0.5,
        )

    def test_find_best_threshold(self, evaluator_for_threshold):
        """Test finding best threshold returns valid result."""
        best_result, df = evaluator_for_threshold.find_best_threshold(metric="f1")
        assert "best_threshold" in best_result
        assert best_result["best_threshold"] >= 0.1
        assert best_result["best_threshold"] <= 0.9
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns

    def test_find_best_threshold_custom_thresholds(self, evaluator_for_threshold):
        """Test finding best threshold with custom threshold list."""
        thresholds = [0.3, 0.5, 0.7]
        best_result, df = evaluator_for_threshold.find_best_threshold(
            metric="f1", thresholds=thresholds
        )
        assert best_result["best_threshold"] in thresholds
        assert len(df) == 3

    def test_find_best_threshold_minimize_metric(self, evaluator_for_threshold):
        """Test finding threshold that minimizes a metric (e.g., smr)."""
        best_result, _ = evaluator_for_threshold.find_best_threshold(metric="smr")
        assert "best_threshold" in best_result
        assert "best_smr" in best_result

    def test_find_best_threshold_invalid_metric(self, evaluator_for_threshold):
        """Test invalid metric raises error."""
        with pytest.raises(ValueError, match="metric must be one of"):
            evaluator_for_threshold.find_best_threshold(metric="invalid_metric")


class TestIntertextEvaluatorQueryIds:
    """Tests for query ID filtering methods."""

    @pytest.fixture
    def evaluator_with_mixed_queries(self, temp_dir):
        """Create evaluator with some queries having matches, some not."""
        query_csv = temp_dir / "query.csv"
        query_csv.write_text(
            "seg_id,text\nq1,Query with match.\nq2,Query without match.\n", encoding="utf-8"
        )
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source.\n", encoding="utf-8")
        gt_csv = temp_dir / "gt.csv"
        gt_csv.write_text(
            "query_id,source_id,label\n"
            "q1,s1,1\n"  # q1 has a match
            "q2,s1,0\n",  # q2 has no match
            encoding="utf-8",
        )

        mock_pipeline = MagicMock()
        s1 = TextSegment("Source.", "s1", row_id=0)
        mock_pipeline.run.return_value = {
            "q1": [CandidateJudge(segment=s1, candidate_score=0.9, judgment_score=0.8)],
            "q2": [CandidateJudge(segment=s1, candidate_score=0.5, judgment_score=0.3)],
        }

        return IntertextEvaluator(
            query_doc=Document(query_csv),
            source_doc=Document(source_csv),
            ground_truth_csv=str(gt_csv),
            pipeline=mock_pipeline,
            threshold=0.5,
        )

    def test_evaluate_with_match_only(self, evaluator_with_mixed_queries):
        """Test evaluating only queries with ground truth labels."""
        df = evaluator_with_mixed_queries.evaluate_all_queries(with_match_only=True)
        # Should include all queries that have ground truth labels
        # (both q1 and q2 have ground truth labels, even though q2 has label=0)
        assert len(df) == 2
        query_ids = set(df["query_id"].tolist())
        assert "q1" in query_ids
        assert "q2" in query_ids
