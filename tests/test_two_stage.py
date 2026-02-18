"""
Unit tests for locisimiles.pipeline.two_stage module.

Tests TwoStagePipeline as a preconfigured Pipeline(EmbeddingCandidateGenerator,
ClassificationJudge).  Deep component behaviour is already covered in
test_generators.py, test_judges.py, and test_pipelines.py.
"""

from unittest.mock import MagicMock, patch

from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.pipeline import Pipeline

# ============== Initialization ==============


class TestTwoStagePipelineInitialization:
    """Tests for two-stage pipeline initialization."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_initialization(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test pipeline initializes with both embedding and classification models."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st_class.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        _pipeline = TwoStagePipeline(device="cpu")

        # Both models should be loaded
        mock_st_class.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_device_configuration(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test device is properly configured."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st_class.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        pipeline = TwoStagePipeline(device="cuda:0")

        assert pipeline.device == "cuda:0"
        mock_model.to.assert_called_with("cuda:0")


# ============== Pipeline composition ==============


class TestTwoStagePipelineComposition:
    """Tests that TwoStagePipeline is correctly composed."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_is_pipeline_subclass(self, mock_tok, mock_mdl, mock_st):
        """TwoStagePipeline should subclass Pipeline."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        m = MagicMock()
        m.to.return_value = m
        m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m

        pipeline = TwoStagePipeline(device="cpu")
        assert isinstance(pipeline, Pipeline)

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_generator_type(self, mock_tok, mock_mdl, mock_st):
        """Generator should be an EmbeddingCandidateGenerator."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        m = MagicMock()
        m.to.return_value = m
        m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m

        pipeline = TwoStagePipeline(device="cpu")
        assert isinstance(pipeline.generator, EmbeddingCandidateGenerator)

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_judge_type(self, mock_tok, mock_mdl, mock_st):
        """Judge should be a ClassificationJudge."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        m = MagicMock()
        m.to.return_value = m
        m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m

        pipeline = TwoStagePipeline(device="cpu")
        assert isinstance(pipeline.judge, ClassificationJudge)

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_pos_class_idx_forwarded(self, mock_tok, mock_mdl, mock_st):
        """pos_class_idx should be forwarded to the ClassificationJudge."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        m = MagicMock()
        m.to.return_value = m
        m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m

        pipeline = TwoStagePipeline(device="cpu", pos_class_idx=0)
        assert pipeline.judge.pos_class_idx == 0

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_caches_are_initially_none(self, mock_tok, mock_mdl, mock_st):
        """Intermediate result caches should be None before run()."""
        from locisimiles.pipeline.two_stage import TwoStagePipeline

        mock_st.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        m = MagicMock()
        m.to.return_value = m
        m.eval.return_value = m
        mock_mdl.from_pretrained.return_value = m

        pipeline = TwoStagePipeline(device="cpu")
        assert pipeline._last_candidates is None
        assert pipeline._last_judgments is None


# ============== Backward compatibility ==============


class TestTwoStagePipelineBackwardCompat:
    """Tests for backward-compatible alias."""

    def test_alias_defined(self):
        """ClassificationPipelineWithCandidategeneration should be an alias."""
        from locisimiles.pipeline.two_stage import (
            ClassificationPipelineWithCandidategeneration,
            TwoStagePipeline,
        )

        assert ClassificationPipelineWithCandidategeneration is TwoStagePipeline

    def test_importable_from_pipeline_package(self):
        """Alias should be importable from the pipeline package."""
        from locisimiles.pipeline import ClassificationPipelineWithCandidategeneration  # noqa

    def test_importable_from_top_level(self):
        """Alias should be importable from the top-level package."""
        from locisimiles import ClassificationPipelineWithCandidategeneration  # noqa
