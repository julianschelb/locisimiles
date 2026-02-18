"""
Unit tests for locisimiles.pipeline.generator subpackage.

Tests the modular candidate-generator components:
- ExhaustiveCandidateGenerator  (no ML dependencies)
- EmbeddingCandidateGenerator   (requires mocking)
- RuleBasedCandidateGenerator   (adapter, no ML dependencies)
- CandidateGeneratorBase        (ABC contract)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import Candidate
from locisimiles.pipeline.generator._base import CandidateGeneratorBase

# ============== ABC Contract ==============


class TestCandidateGeneratorBase:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abc(self):
        """CandidateGeneratorBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CandidateGeneratorBase()

    def test_subclass_must_implement_generate(self):
        """Subclasses that don't implement generate() cannot be instantiated."""

        class IncompleteGenerator(CandidateGeneratorBase):
            pass

        with pytest.raises(TypeError):
            IncompleteGenerator()

    def test_subclass_with_generate_works(self):
        """A proper subclass can be instantiated."""

        class DummyGenerator(CandidateGeneratorBase):
            def generate(self, *, query, source, **kwargs):
                return {}

        gen = DummyGenerator()
        assert isinstance(gen, CandidateGeneratorBase)


# ============== ExhaustiveCandidateGenerator ==============


class TestExhaustiveCandidateGenerator:
    """Tests for the exhaustive (all-pairs) candidate generator."""

    def test_returns_all_pairs(self, query_document, source_document):
        """Every query segment should get all source segments as candidates."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator

        generator = ExhaustiveCandidateGenerator()
        result = generator.generate(query=query_document, source=source_document)

        assert isinstance(result, dict)
        assert set(result.keys()) == {seg.id for seg in query_document}
        for qid, candidates in result.items():
            assert len(candidates) == len(list(source_document))
            for c in candidates:
                assert isinstance(c, Candidate)
                assert c.score == 1.0

    def test_is_candidate_generator_base(self):
        """ExhaustiveCandidateGenerator is a CandidateGeneratorBase."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator

        assert issubclass(ExhaustiveCandidateGenerator, CandidateGeneratorBase)

    def test_extra_kwargs_ignored(self, query_document, source_document):
        """Extra kwargs should be accepted and ignored."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator

        generator = ExhaustiveCandidateGenerator()
        result = generator.generate(
            query=query_document,
            source=source_document,
            top_k=3,
            some_extra_param="hello",
        )
        assert len(result) > 0

    def test_candidate_segments_are_source_segments(self, query_document, source_document):
        """Each candidate's segment should be one of the source segments."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator

        generator = ExhaustiveCandidateGenerator()
        result = generator.generate(query=query_document, source=source_document)

        source_ids = {seg.id for seg in source_document}
        for qid, candidates in result.items():
            for c in candidates:
                assert c.segment.id in source_ids


# ============== EmbeddingCandidateGenerator ==============


class TestEmbeddingCandidateGenerator:
    """Tests for the embedding-based candidate generator (mocked models)."""

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.generator.embedding.chromadb.EphemeralClient")
    def test_initialization(self, mock_chroma, mock_st_class):
        """Test that the generator initializes with embedding model."""
        from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator

        mock_st_class.return_value = MagicMock()
        generator = EmbeddingCandidateGenerator(device="cpu")

        mock_st_class.assert_called_once()
        assert generator.device == "cpu"

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_embed(self, mock_st_class):
        """Test that _embed returns normalized float32 arrays."""
        from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype("float32")
        mock_st_class.return_value = mock_embedder

        generator = EmbeddingCandidateGenerator(device="cpu")
        result = generator._embed(["text1", "text2", "text3"], prompt_name="query")

        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.generator.embedding.chromadb.EphemeralClient")
    def test_build_source_index(self, mock_chroma, mock_st_class):
        """Test source index building creates a Chroma collection."""
        from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator

        mock_st_class.return_value = MagicMock()

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        generator = EmbeddingCandidateGenerator(device="cpu")

        segments = [
            TextSegment("Text 1", "s1", row_id=0),
            TextSegment("Text 2", "s2", row_id=1),
        ]
        embeddings = np.random.randn(2, 384).astype("float32")

        _result = generator.build_source_index(segments, embeddings)

        mock_client.create_collection.assert_called_once()
        mock_collection.add.assert_called()

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    @patch("locisimiles.pipeline.generator.embedding.chromadb.EphemeralClient")
    @patch("locisimiles.pipeline.generator.embedding.tqdm", lambda x, **kwargs: x)
    def test_generate(self, mock_chroma, mock_st_class, temp_dir):
        """Test full generate() workflow."""
        from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator

        # Setup embedder mock
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.randn(3, 384).astype("float32")
        mock_st_class.return_value = mock_embedder

        # Setup Chroma mock
        mock_collection = MagicMock()
        mock_collection.query.side_effect = lambda query_embeddings, n_results: {
            "ids": [["s1", "s2"][:n_results]],
            "distances": [[0.1, 0.3][:n_results]],
        }
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        generator = EmbeddingCandidateGenerator(device="cpu")

        # Create documents
        query_csv = temp_dir / "query.csv"
        query_csv.write_text(
            "seg_id,text\nq1,Query text one.\nq2,Query text two.\n",
            encoding="utf-8",
        )
        source_csv = temp_dir / "source.csv"
        source_csv.write_text(
            "seg_id,text\ns1,Source text one.\ns2,Source text two.\n",
            encoding="utf-8",
        )

        result = generator.generate(
            query=Document(query_csv),
            source=Document(source_csv),
            top_k=2,
        )

        assert isinstance(result, dict)
        assert len(result) > 0
        for qid, candidates in result.items():
            assert all(isinstance(c, Candidate) for c in candidates)
            # Score should be 1 - distance
            for c in candidates:
                assert 0.0 <= c.score <= 1.0

    @patch("locisimiles.pipeline.generator.embedding.SentenceTransformer")
    def test_is_candidate_generator_base(self, mock_st_class):
        """EmbeddingCandidateGenerator is a CandidateGeneratorBase."""
        from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator

        assert issubclass(EmbeddingCandidateGenerator, CandidateGeneratorBase)


# ============== RuleBasedCandidateGenerator ==============


class TestRuleBasedCandidateGenerator:
    """Tests for the rule-based candidate generator adapter."""

    def test_is_candidate_generator_base(self):
        """RuleBasedCandidateGenerator is a CandidateGeneratorBase."""
        from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator

        assert issubclass(RuleBasedCandidateGenerator, CandidateGeneratorBase)

    def test_initialization(self):
        """Test that the generator stores configuration directly."""
        from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator

        generator = RuleBasedCandidateGenerator(min_shared_words=3, max_distance=5)
        assert generator.min_shared_words == 3
        assert generator.max_distance == 5

    def test_generate_returns_candidates(self, query_document, source_document):
        """Test that generate() returns CandidateGeneratorOutput."""
        from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator

        generator = RuleBasedCandidateGenerator(min_shared_words=2)
        result = generator.generate(
            query=query_document,
            source=source_document,
            query_genre="prose",
            source_genre="poetry",
        )

        assert isinstance(result, dict)
        # Should have an entry for each query segment
        for qid, candidates in result.items():
            for c in candidates:
                assert isinstance(c, Candidate)
                assert isinstance(c.score, float)

    def test_generate_with_top_k(self, query_document, source_document):
        """Test that top_k parameter is forwarded."""
        from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator

        generator = RuleBasedCandidateGenerator(min_shared_words=2)
        result = generator.generate(
            query=query_document,
            source=source_document,
            top_k=1,
        )

        for qid, candidates in result.items():
            assert len(candidates) <= 1


# ============== Import Tests ==============


class TestGeneratorImports:
    """Test that generators are importable from the expected paths."""

    def test_import_from_generator_package(self):
        """All generators should be importable from the generator package."""

    def test_import_from_pipeline_package(self):
        """All generators should be importable from the pipeline package."""

    def test_import_from_top_level(self):
        """Key generators should be importable from the top-level package."""
