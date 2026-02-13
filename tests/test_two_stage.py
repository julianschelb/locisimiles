"""
Unit tests for locisimiles.pipeline.two_stage module.
Tests ClassificationPipelineWithCandidategeneration class with mocked ML models.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import FullDict, SimDict


class TestTwoStagePipelineInitialization:
    """Tests for two-stage pipeline initialization."""

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    def test_initialization(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test pipeline initializes with both embedding and classification models."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_st_class.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        
        # Both models should be loaded
        mock_st_class.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    def test_device_configuration(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test device is properly configured."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_st_class.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cuda:0")
        
        assert pipeline.device == "cuda:0"
        # Classifier moved to device
        mock_model.to.assert_called_with("cuda:0")


class TestTwoStagePipelineRetrieval:
    """Tests for the retrieval stage of the two-stage pipeline."""

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    def test_build_source_index(self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test source index building."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_st_class.return_value = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        
        segments = [
            TextSegment("Text 1", "s1", row_id=0),
            TextSegment("Text 2", "s2", row_id=1),
        ]
        embeddings = np.random.randn(2, 384).astype("float32")
        
        result = pipeline.build_source_index(segments, embeddings)
        
        mock_client.create_collection.assert_called_once()
        mock_collection.add.assert_called()


class TestTwoStagePipelineClassification:
    """Tests for the classification stage of the two-stage pipeline."""

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    def test_truncate_pair(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test text pair truncation."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_st_class.return_value = MagicMock()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()[:250]
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        
        long_text = " ".join(["word"] * 300)
        trunc1, trunc2 = pipeline._truncate_pair(long_text, long_text)
        
        # Should be truncated
        assert len(trunc1.split()) <= 255
        assert len(trunc2.split()) <= 255

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    def test_predict_batch(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test batch prediction."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_st_class.return_value = MagicMock()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(3, 2)
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        
        probs = pipeline._predict_batch("query", ["c1", "c2", "c3"])
        
        assert len(probs) == 3
        for p in probs:
            assert 0.0 <= p <= 1.0


class TestTwoStagePipelineRun:
    """Tests for the full run() method of the two-stage pipeline."""

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.two_stage.tqdm", lambda x, **kwargs: x)
    def test_run_retrieval_then_classification(
        self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class, temp_dir
    ):
        """Test that pipeline runs both retrieval and classification stages."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        # Setup embedder mock
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup classifier mock
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(2, 2)  # top_k=2
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Setup chroma mock
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1", "s2"]],
            "distances": [[0.1, 0.3]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Create documents
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query text\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,Source 1\ns2,Source 2\ns3,Source 3\n",
            encoding="utf-8"
        )
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=2,
        )
        
        # Should have results for q1
        assert "q1" in result
        # Should have 2 candidates (top_k=2)
        assert len(result["q1"]) == 2

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.two_stage.tqdm", lambda x, **kwargs: x)
    def test_run_top_k_candidates(
        self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class, temp_dir
    ):
        """Test that only top-k candidates are processed."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(5, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(3, 2)  # top_k=3
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1", "s2", "s3"]],  # top 3
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,S1\ns2,S2\ns3,S3\ns4,S4\ns5,S5\n",
            encoding="utf-8"
        )
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=3,
        )
        
        # Only top-3 candidates should be in results
        assert len(result["q1"]) == 3

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.two_stage.tqdm", lambda x, **kwargs: x)
    def test_run_similarity_in_results(
        self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class, temp_dir
    ):
        """Test that similarity scores are preserved in output."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(1, 2)
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_collection = MagicMock()
        # Distance 0.2 → similarity 0.8
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.2]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=1,
        )
        
        segment, similarity, prob = result["q1"][0]
        assert similarity is not None
        assert similarity == pytest.approx(0.8, rel=1e-5)

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.two_stage.tqdm", lambda x, **kwargs: x)
    def test_run_probability_in_results(
        self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class, temp_dir
    ):
        """Test that classification probabilities are in output."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        # Create logits that will produce known probability
        mock_result.logits = torch.tensor([[0.0, 2.0]])  # High prob for class 1
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.1]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=1,
        )
        
        segment, similarity, prob = result["q1"][0]
        assert 0.0 <= prob <= 1.0
        # With logits [0, 2], softmax gives ~0.88 for class 1
        assert prob > 0.8

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.two_stage.tqdm", lambda x, **kwargs: x)
    def test_fulldict_format(
        self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class, temp_dir
    ):
        """Test output matches FullDict type structure."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(1, 2)
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.1]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=1,
        )
        
        # Verify FullDict structure
        assert isinstance(result, dict)
        for qid, pairs in result.items():
            assert isinstance(qid, str)
            for segment, similarity, probability in pairs:
                assert isinstance(segment, TextSegment)
                assert isinstance(similarity, float)
                assert isinstance(probability, float)


class TestTwoStagePipelineDebug:
    """Tests for debug methods."""

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    def test_debug_input_sequence(self, mock_tokenizer_class, mock_model_class, mock_st_class):
        """Test debug method returns expected keys."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_st_class.return_value = MagicMock()
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_encoding.__getitem__ = MagicMock(return_value=torch.zeros(1, 512))
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer.decode.return_value = "[CLS] query [SEP] candidate [SEP]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        
        result = pipeline.debug_input_sequence("query text", "candidate text")
        
        expected_keys = [
            "query", "candidate", "query_truncated", "candidate_truncated",
            "input_ids", "attention_mask", "input_text"
        ]
        for key in expected_keys:
            assert key in result


class TestTwoStagePipelineResultCaching:
    """Tests for result caching behavior."""

    @patch("locisimiles.pipeline.two_stage.SentenceTransformer")
    @patch("locisimiles.pipeline.two_stage.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.two_stage.AutoTokenizer")
    @patch("locisimiles.pipeline.two_stage.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.two_stage.tqdm", lambda x, **kwargs: x)
    def test_stores_last_sim_and_full(
        self, mock_chroma, mock_tokenizer_class, mock_model_class, mock_st_class, temp_dir
    ):
        """Test that _last_sim and _last_full are populated after run."""
        from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(1, 2)
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.1]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")
        
        # Before run
        assert pipeline._last_sim is None
        assert pipeline._last_full is None
        
        pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=1,
        )
        
        # After run
        assert pipeline._last_sim is not None
        assert pipeline._last_full is not None
