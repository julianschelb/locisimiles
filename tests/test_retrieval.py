"""
Unit tests for locisimiles.pipeline.retrieval module.

Tests RetrievalPipeline as a preconfigured
Pipeline(EmbeddingCandidateGenerator, ThresholdJudge).
Deep component behaviour is already covered in test_generators.py,
test_judges.py, and test_pipelines.py.
"""

from unittest.mock import MagicMock, patch

from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.pipeline import Pipeline

# ============== Initialization ==============


class TestRetrievalPipelineInitialization:
    """Tests for retrieval pipeline initialization."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_initialization(self, mock_st_class):
        """Test pipeline initializes with embedding model."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        _pipeline = RetrievalPipeline(device="cpu")

        mock_st_class.assert_called_once()

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_device_configuration(self, mock_st_class):
        """Test device is properly configured."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cuda:0")

        assert pipeline.device == "cuda:0"

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_default_device_cpu(self, mock_st_class):
        """Test default device is CPU."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline()

        assert pipeline.device == "cpu"


# ============== Pipeline composition ==============


class TestRetrievalPipelineComposition:
    """Tests that RetrievalPipeline is correctly composed."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_is_pipeline_subclass(self, mock_st_class):
        """Should subclass Pipeline."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu")
        assert isinstance(pipeline, Pipeline)

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_generator_type(self, mock_st_class):
        """Generator should be EmbeddingCandidateGenerator."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu")
        assert isinstance(pipeline.generator, EmbeddingCandidateGenerator)

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_judge_type(self, mock_st_class):
        """Judge should be ThresholdJudge."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu")
        assert isinstance(pipeline.judge, ThresholdJudge)

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_top_k_forwarded(self, mock_st_class):
        """top_k should be forwarded to the ThresholdJudge."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu", top_k=5)
        assert pipeline.judge.top_k == 5

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_similarity_threshold_forwarded(self, mock_st_class):
        """similarity_threshold should be forwarded to the ThresholdJudge."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu", similarity_threshold=0.75)
        assert pipeline.judge.similarity_threshold == 0.75

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_caches_are_initially_none(self, mock_st_class):
        """Intermediate result caches should be None before run()."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline

        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu")
        assert pipeline._last_candidates is None
        assert pipeline._last_judgments is None
