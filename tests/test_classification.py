"""
Unit tests for locisimiles.pipeline.classification module.

Tests ExhaustiveClassificationPipeline as a preconfigured
Pipeline(ExhaustiveCandidateGenerator, ClassificationJudge).
Deep component behaviour is already covered in test_generators.py,
test_judges.py, and test_pipelines.py.
"""
import pytest
from unittest.mock import patch, MagicMock

from locisimiles.pipeline.pipeline import Pipeline
from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge


# ============== Initialization ==============


class TestClassificationPipelineInitialization:
    """Tests for classification pipeline initialization."""

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_initialization(self, mock_tok, mock_mdl):
        """Test pipeline initializes with classification model."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu")

        mock_tok.from_pretrained.assert_called_once()
        mock_mdl.from_pretrained.assert_called_once()

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_device_configuration(self, mock_tok, mock_mdl):
        """Test device is properly configured."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cuda:0")

        assert pipeline.device == "cuda:0"
        m.to.assert_called_with("cuda:0")

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_default_device_cpu(self, mock_tok, mock_mdl):
        """Test default device is CPU."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline()

        assert pipeline.device == "cpu"


# ============== Pipeline composition ==============


class TestClassificationPipelineComposition:
    """Tests that ExhaustiveClassificationPipeline is correctly composed."""

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_is_pipeline_subclass(self, mock_tok, mock_mdl):
        """Should subclass Pipeline."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu")
        assert isinstance(pipeline, Pipeline)

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_generator_type(self, mock_tok, mock_mdl):
        """Generator should be ExhaustiveCandidateGenerator."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu")
        assert isinstance(pipeline.generator, ExhaustiveCandidateGenerator)

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_judge_type(self, mock_tok, mock_mdl):
        """Judge should be ClassificationJudge."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu")
        assert isinstance(pipeline.judge, ClassificationJudge)

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_pos_class_idx_forwarded(self, mock_tok, mock_mdl):
        """pos_class_idx should be forwarded to the ClassificationJudge."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu", pos_class_idx=0)
        assert pipeline.judge.pos_class_idx == 0

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_default_pos_class_idx(self, mock_tok, mock_mdl):
        """Default positive class index should be 1."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu")
        assert pipeline.judge.pos_class_idx == 1

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_caches_are_initially_none(self, mock_tok, mock_mdl):
        """Intermediate result caches should be None before run()."""
        from locisimiles.pipeline.classification import ExhaustiveClassificationPipeline

        m = MagicMock(); m.to.return_value = m; m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m
        mock_tok.from_pretrained.return_value = MagicMock()

        pipeline = ExhaustiveClassificationPipeline(device="cpu")
        assert pipeline._last_candidates is None
        assert pipeline._last_judgments is None


# ============== Backward compatibility ==============


class TestClassificationPipelineBackwardCompat:
    """Tests for backward-compatible alias."""

    def test_alias_defined(self):
        """ClassificationPipeline should be an alias."""
        from locisimiles.pipeline.classification import (
            ExhaustiveClassificationPipeline,
            ClassificationPipeline,
        )
        assert ClassificationPipeline is ExhaustiveClassificationPipeline

    def test_importable_from_pipeline_package(self):
        """Alias should be importable from the pipeline package."""
        from locisimiles.pipeline import ClassificationPipeline  # noqa

    def test_importable_from_top_level(self):
        """Alias should be importable from the top-level package."""
        from locisimiles import ClassificationPipeline  # noqa
