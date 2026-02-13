"""
Unit tests for locisimiles.pipeline.retrieval module.
Tests RetrievalPipeline class with mocked ML models.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import SimDict, FullDict


class TestRetrievalPipelineEmbedding:
    """Tests for embedding generation in RetrievalPipeline."""

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_embed_returns_normalized(self, mock_st_class):
        """Test that _embed returns normalized vectors."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        # Setup mock
        mock_embedder = MagicMock()
        # Return non-normalized embeddings
        raw_embeddings = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        mock_embedder.encode.return_value = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        mock_st_class.return_value = mock_embedder
        
        pipeline = RetrievalPipeline(device="cpu")
        result = pipeline._embed(["text1", "text2"], prompt_name="query")
        
        # Check normalization (each row should have norm ~1)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=5)

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_embed_shape(self, mock_st_class):
        """Test that embedding shape matches input count."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_embedder = MagicMock()
        # Return embeddings with correct shape
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        pipeline = RetrievalPipeline(device="cpu")
        result = pipeline._embed(["t1", "t2", "t3"], prompt_name="query")
        
        assert result.shape[0] == 3
        assert len(result.shape) == 2

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_embed_dtype(self, mock_st_class):
        """Test that embeddings are float32."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        pipeline = RetrievalPipeline(device="cpu")
        result = pipeline._embed(["t1", "t2"], prompt_name="match")
        
        assert result.dtype == np.float32


class TestRetrievalPipelineIndexBuilding:
    """Tests for index building in RetrievalPipeline."""

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    def test_build_source_index(self, mock_chroma_client, mock_st_class):
        """Test index creation with segments and embeddings."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        # Setup mocks
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        mock_st_class.return_value = MagicMock()
        
        pipeline = RetrievalPipeline(device="cpu")
        
        segments = [
            TextSegment("Text 1", "s1", row_id=0),
            TextSegment("Text 2", "s2", row_id=1),
        ]
        embeddings = np.random.randn(2, 384).astype("float32")
        
        result = pipeline.build_source_index(segments, embeddings)
        
        # Verify collection was created
        mock_client_instance.create_collection.assert_called_once()
        # Verify segments were added
        mock_collection.add.assert_called()

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    def test_build_source_index_batching(self, mock_chroma_client, mock_st_class):
        """Test index building with batched adds for large datasets."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        mock_st_class.return_value = MagicMock()
        
        pipeline = RetrievalPipeline(device="cpu")
        
        # Create more segments than batch size
        num_segments = 150
        segments = [TextSegment(f"Text {i}", f"s{i}", row_id=i) for i in range(num_segments)]
        embeddings = np.random.randn(num_segments, 384).astype("float32")
        
        pipeline.build_source_index(segments, embeddings, batch_size=100)
        
        # Should have 2 batch calls (100 + 50)
        assert mock_collection.add.call_count == 2


class TestRetrievalPipelineSimilarity:
    """Tests for similarity computation in RetrievalPipeline."""

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_compute_similarity_returns_simdict(self, mock_st_class):
        """Test that _compute_similarity returns proper SimDict structure."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu")
        
        # Setup mock index
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "ids": [["s1", "s2"]],
            "distances": [[0.1, 0.3]],  # Cosine distances
        }
        pipeline._source_index = mock_index
        
        # Create source document
        source_doc = MagicMock()
        source_doc.__getitem__ = lambda self, x: TextSegment(f"Text {x}", x, row_id=0)
        
        query_segments = [TextSegment("Query", "q1", row_id=0)]
        query_embeddings = np.random.randn(1, 384).astype("float32")
        
        result = pipeline._compute_similarity(
            query_segments, query_embeddings, source_doc, top_k=2
        )
        
        assert isinstance(result, dict)
        assert "q1" in result
        assert len(result["q1"]) == 2

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_similarity_score_range(self, mock_st_class):
        """Test that similarity scores are converted from distance correctly."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cpu")
        
        # Setup mock index with known distances
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.2]],  # Distance 0.2 → similarity 0.8
        }
        pipeline._source_index = mock_index
        
        source_doc = MagicMock()
        source_doc.__getitem__ = lambda self, x: TextSegment("Text", x, row_id=0)
        
        query_segments = [TextSegment("Query", "q1", row_id=0)]
        query_embeddings = np.random.randn(1, 384).astype("float32")
        
        result = pipeline._compute_similarity(
            query_segments, query_embeddings, source_doc, top_k=1
        )
        
        _, similarity = result["q1"][0]
        assert similarity == pytest.approx(0.8, rel=1e-5)


class TestRetrievalPipelineRun:
    """Tests for the main run() method of RetrievalPipeline."""

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.retrieval.tqdm", lambda x, **kwargs: x)
    def test_run_topk_mode(self, mock_chroma_client, mock_st_class, temp_dir):
        """Test that top-k candidates are marked as positive."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1", "s2", "s3"]],
            "distances": [[0.1, 0.3, 0.5]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        
        # Create documents
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query text\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,Source1\ns2,Source2\ns3,Source3\n",
            encoding="utf-8"
        )
        
        pipeline = RetrievalPipeline(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=2,
        )
        
        assert isinstance(result, dict)
        assert "q1" in result
        # Top 2 should have prob=1.0, third should have prob=0.0
        probs = [prob for _, _, prob in result["q1"]]
        assert probs[0] == 1.0
        assert probs[1] == 1.0
        assert probs[2] == 0.0

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.retrieval.tqdm", lambda x, **kwargs: x)
    def test_run_threshold_mode(self, mock_chroma_client, mock_st_class, temp_dir):
        """Test threshold-based positive selection."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_collection = MagicMock()
        # Distances: 0.1 → sim=0.9, 0.3 → sim=0.7, 0.8 → sim=0.2
        mock_collection.query.return_value = {
            "ids": [["s1", "s2", "s3"]],
            "distances": [[0.1, 0.3, 0.8]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,S1\ns2,S2\ns3,S3\n",
            encoding="utf-8"
        )
        
        pipeline = RetrievalPipeline(device="cpu")
        result = pipeline.run(
            query=Document(query_csv),
            source=Document(source_csv),
            similarity_threshold=0.5,  # Only s1 (0.9) and s2 (0.7) should be positive
        )
        
        probs = [prob for _, _, prob in result["q1"]]
        # sim >= 0.5: s1=1.0, s2=1.0
        # sim < 0.5: s3=0.0
        assert probs[0] == 1.0  # sim=0.9
        assert probs[1] == 1.0  # sim=0.7
        assert probs[2] == 0.0  # sim=0.2

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.retrieval.tqdm", lambda x, **kwargs: x)
    def test_fulldict_format(self, mock_chroma_client, mock_st_class, temp_dir):
        """Test output matches FullDict type structure."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.2]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = RetrievalPipeline(device="cpu")
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
                assert 0.0 <= probability <= 1.0


class TestRetrievalPipelineRetrieve:
    """Tests for the retrieve() method."""

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.retrieval.tqdm", lambda x, **kwargs: x)
    def test_retrieve_returns_simdict(self, mock_chroma_client, mock_st_class, temp_dir):
        """Test that retrieve() returns SimDict format."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1", "s2"]],
            "distances": [[0.1, 0.2]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,S1\ns2,S2\n", encoding="utf-8")
        
        pipeline = RetrievalPipeline(device="cpu")
        result = pipeline.retrieve(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=2,
        )
        
        # SimDict: Dict[str, List[Tuple[TextSegment, float]]]
        assert isinstance(result, dict)
        for qid, pairs in result.items():
            for segment, score in pairs:
                assert isinstance(segment, TextSegment)
                assert isinstance(score, float)
                # Should only have 2 values (no probability)

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    @patch("locisimiles.pipeline.retrieval.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.retrieval.tqdm", lambda x, **kwargs: x)
    def test_retrieve_stores_last_sim(self, mock_chroma_client, mock_st_class, temp_dir):
        """Test that retrieve() stores results in _last_sim."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(2, 384).astype("float32")
        mock_st_class.return_value = mock_embedder
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["s1"]],
            "distances": [[0.1]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance
        
        query_csv = temp_dir / "query.csv"
        query_csv.write_text("seg_id,text\nq1,Query\n", encoding="utf-8")
        source_csv = temp_dir / "source.csv"
        source_csv.write_text("seg_id,text\ns1,Source\n", encoding="utf-8")
        
        pipeline = RetrievalPipeline(device="cpu")
        assert pipeline._last_sim is None
        
        pipeline.retrieve(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=1,
        )
        
        assert pipeline._last_sim is not None


class TestRetrievalPipelineDeviceSelection:
    """Tests for device selection in RetrievalPipeline."""

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_default_device_cpu(self, mock_st_class):
        """Test default device is CPU."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline()
        
        assert pipeline.device == "cpu"

    @patch("locisimiles.pipeline.retrieval.SentenceTransformer")
    def test_custom_device(self, mock_st_class):
        """Test custom device is set."""
        from locisimiles.pipeline.retrieval import RetrievalPipeline
        
        mock_st_class.return_value = MagicMock()
        pipeline = RetrievalPipeline(device="cuda")
        
        assert pipeline.device == "cuda"
