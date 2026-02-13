"""
Unit tests for locisimiles.pipeline.classification module.
Tests ClassificationPipeline class with mocked ML models.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import FullDict


class TestClassificationPipelineTruncation:
    """Tests for text truncation in ClassificationPipeline."""

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_truncate_pair_long_texts(self, mock_tokenizer_class, mock_model_class):
        """Test that long texts are truncated to fit max_len."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        # Simulate tokenization
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()[:250]
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cpu")
        
        # Create long sentence
        long_sentence = " ".join(["word"] * 300)
        
        trunc1, trunc2 = pipeline._truncate_pair(long_sentence, long_sentence, max_len=512)
        
        # Each should be truncated
        assert len(trunc1.split()) <= 255  # (512 - 3) // 2 = 254.5
        assert len(trunc2.split()) <= 255

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_truncate_pair_short_texts(self, mock_tokenizer_class, mock_model_class):
        """Test that short texts pass through without truncation."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cpu")
        
        short1 = "This is short"
        short2 = "Also short"
        
        trunc1, trunc2 = pipeline._truncate_pair(short1, short2)
        
        assert trunc1 == short1
        assert trunc2 == short2

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_count_special_tokens(self, mock_tokenizer_class, mock_model_class):
        """Test counting special tokens for pair encoding."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 4  # e.g., [CLS] + 2x [SEP] + padding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cpu")
        
        count = pipeline._count_special_tokens_added()
        assert count == 4
        mock_tokenizer.num_special_tokens_to_add.assert_called_with(pair=True)


class TestClassificationPipelinePrediction:
    """Tests for prediction methods in ClassificationPipeline."""

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_predict_batch_shape(self, mock_tokenizer_class, mock_model_class):
        """Test that batch prediction returns correct number of probabilities."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        
        # Return proper encoding mock
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_encoding.__getitem__ = lambda self, key: torch.zeros(3, 512)
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        # Return logits for 3 samples
        mock_result = MagicMock()
        mock_result.logits = torch.randn(3, 2)
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cpu")
        
        query_text = "Query sentence"
        cand_texts = ["Candidate 1", "Candidate 2", "Candidate 3"]
        
        probs = pipeline._predict_batch(query_text, cand_texts)
        
        assert len(probs) == 3

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_predict_probability_range(self, mock_tokenizer_class, mock_model_class):
        """Test that probabilities are in [0, 1] range."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
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
        # Create logits that will produce probabilities after softmax
        mock_result.logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cpu")
        
        probs = pipeline._predict_batch("query", ["cand1", "cand2"])
        
        for prob in probs:
            assert 0.0 <= prob <= 1.0

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_predict_batching(self, mock_tokenizer_class, mock_model_class):
        """Test that _predict handles batching correctly."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        
        # Track batch sizes for proper mock response
        batch_sizes = []
        def create_mock_encoding(*args, **kwargs):
            batch_size = len(args[0]) if args else 1
            batch_sizes.append(batch_size)
            mock_enc = MagicMock()
            mock_enc.to.return_value = mock_enc
            return mock_enc
        mock_tokenizer.side_effect = create_mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # Return correct batch sizes based on actual input
        batch_idx = [0]
        def mock_forward(**kwargs):
            result = MagicMock()
            # Return logits matching the batch size
            bs = batch_sizes[batch_idx[0]] if batch_idx[0] < len(batch_sizes) else 2
            batch_idx[0] += 1
            result.logits = torch.randn(bs, 2)
            return result
        mock_model.side_effect = mock_forward
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cpu")
        
        # 5 candidates with batch_size=2 → 3 batches (2+2+1)
        cand_texts = [f"Candidate {i}" for i in range(5)]
        probs = pipeline._predict("query", cand_texts, batch_size=2)
        
        assert len(probs) == 5


class TestClassificationPipelineRun:
    """Tests for the main run() method of ClassificationPipeline."""

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    @patch("locisimiles.pipeline.classification.tqdm", lambda x, **kwargs: x)
    def test_run_all_pairs(self, mock_tokenizer_class, mock_model_class, temp_dir):
        """Test that all query-source pairs are classified."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
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
        
        def mock_forward(**kwargs):
            result = MagicMock()
            result.logits = torch.randn(3, 2)  # 3 source segments
            return result
        mock_model.side_effect = mock_forward
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create documents
        query_csv = temp_dir / "query.csv"
        query_csv.write_text(
            "seg_id,text\nq1,Query 1\nq2,Query 2\n",
            encoding="utf-8"
        )
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,Source 1\ns2,Source 2\ns3,Source 3\n",
            encoding="utf-8"
        )
        
        pipeline = ClassificationPipeline(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
        )
        
        # Should have 2 queries, each with 3 source candidates
        assert len(result) == 2
        assert "q1" in result
        assert "q2" in result
        assert len(result["q1"]) == 3
        assert len(result["q2"]) == 3

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    @patch("locisimiles.pipeline.classification.tqdm", lambda x, **kwargs: x)
    def test_run_none_similarity(self, mock_tokenizer_class, mock_model_class, temp_dir):
        """Test that similarity is None (no retrieval stage)."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
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
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = ClassificationPipeline(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
        )
        
        segment, similarity, prob = result["q1"][0]
        assert similarity is None  # No retrieval stage


class TestClassificationPipelineDebug:
    """Tests for debug methods in ClassificationPipeline."""

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_debug_input_sequence(self, mock_tokenizer_class, mock_model_class):
        """Test debug method returns expected keys."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
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
        
        pipeline = ClassificationPipeline(device="cpu")
        
        result = pipeline.debug_input_sequence("query text", "candidate text")
        
        expected_keys = [
            "query", "candidate", "query_truncated", "candidate_truncated",
            "input_ids", "attention_mask", "input_text"
        ]
        for key in expected_keys:
            assert key in result


class TestClassificationPipelineDeviceSelection:
    """Tests for device selection in ClassificationPipeline."""

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_default_device_cpu(self, mock_tokenizer_class, mock_model_class):
        """Test default device is CPU."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline()
        
        assert pipeline.device == "cpu"

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_custom_device(self, mock_tokenizer_class, mock_model_class):
        """Test custom device is set and model moved."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(device="cuda:0")
        
        assert pipeline.device == "cuda:0"
        mock_model.to.assert_called_with("cuda:0")


class TestClassificationPipelinePositiveClassIndex:
    """Tests for positive class index configuration."""

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_default_pos_class_idx(self, mock_tokenizer_class, mock_model_class):
        """Test default positive class index is 1."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline()
        
        assert pipeline.pos_class_idx == 1

    @patch("locisimiles.pipeline.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.classification.AutoTokenizer")
    def test_custom_pos_class_idx(self, mock_tokenizer_class, mock_model_class):
        """Test custom positive class index."""
        from locisimiles.pipeline.classification import ClassificationPipeline
        
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        pipeline = ClassificationPipeline(pos_class_idx=0)
        
        assert pipeline.pos_class_idx == 0
