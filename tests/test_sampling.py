"""Tests for the standalone negative-sampling functions.

``sample_hard_negatives`` is tested against a mocked
``EmbeddingCandidateGenerator`` (matching the ``mock_embedder``-style
convention already used elsewhere in this suite) rather than downloading a
real embedding model, to keep the suite fast.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from locisimiles.ground_truth import GroundTruth
from locisimiles.pipeline._types import Candidate
from locisimiles.training.sampling import (
    sample_hard_negatives,
    sample_random_negatives,
    sample_random_pairs,
)


@pytest.fixture
def positives() -> GroundTruth:
    return GroundTruth([{"query_id": "q1", "source_id": "s1", "label": "cit"}])


class TestSampleRandomPairs:
    def test_never_returns_a_known_positive(self, query_document, source_document, positives):
        result = sample_random_pairs(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=5,
            seed=1,
        )
        known = {(e.query_id, e.source_id) for e in positives}
        for entry in result:
            assert (entry.query_id, entry.source_id) not in known
            assert entry.label == "no_match"

    def test_respects_n_per_query_target_count(self, query_document, source_document, positives):
        result = sample_random_pairs(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=2,
            seed=1,
        )
        assert len(result) <= 2 * len(query_document)

    def test_reproducible_for_fixed_seed(self, query_document, source_document, positives):
        kwargs = dict(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=3,
            seed=11,
        )
        first = sample_random_pairs(**kwargs)
        second = sample_random_pairs(**kwargs)
        assert [(e.query_id, e.source_id) for e in first] == [
            (e.query_id, e.source_id) for e in second
        ]

    def test_not_restricted_to_queries_with_positives(self, query_document, source_document):
        """Unlike the other sampling methods, pairs aren't conditioned per query."""
        result = sample_random_pairs(
            query_doc=query_document,
            source_doc=source_document,
            positives=GroundTruth(),
            n_per_query=10,
            seed=2,
        )
        assert len({e.query_id for e in result}) > 1

    def test_returns_ground_truth_concatenable(self, query_document, source_document, positives):
        result = sample_random_pairs(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=1,
            seed=1,
        )
        combined = positives + result
        assert len(combined) == len(positives) + len(result)


class TestSampleRandomNegatives:
    def test_every_query_gets_n_per_query_negatives(
        self, query_document, source_document, positives
    ):
        result = sample_random_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=2,
            seed=42,
        )
        by_query: dict = {}
        for entry in result:
            by_query.setdefault(entry.query_id, []).append(entry)
        assert set(by_query) == {seg.id for seg in query_document}
        assert all(len(v) == 2 for v in by_query.values())

    def test_never_returns_a_known_positive(self, query_document, source_document, positives):
        result = sample_random_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=4,
            seed=1,
        )
        known = {(e.query_id, e.source_id) for e in positives}
        for entry in result:
            assert (entry.query_id, entry.source_id) not in known
            assert entry.label == "no_match"

    def test_reproducible_for_fixed_seed(self, query_document, source_document, positives):
        kwargs = dict(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=2,
            seed=5,
        )
        first = sample_random_negatives(**kwargs)
        second = sample_random_negatives(**kwargs)
        assert [(e.query_id, e.source_id) for e in first] == [
            (e.query_id, e.source_id) for e in second
        ]

    def test_custom_label(self, query_document, source_document, positives):
        result = sample_random_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=1,
            seed=1,
            label=0,
        )
        assert all(entry.label == 0 for entry in result)


class TestSampleHardNegatives:
    @patch("locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator")
    def test_excludes_known_positives_even_when_top_ranked(
        self, mock_generator_cls, query_document, source_document, positives
    ):
        segments = {seg.id: seg for seg in source_document}
        instance = MagicMock()
        instance.generate.return_value = {
            "q1": [
                Candidate(segment=segments["s1"], score=0.99),  # known positive, must be skipped
                Candidate(segment=segments["s2"], score=0.80),
                Candidate(segment=segments["s3"], score=0.50),
            ],
            "q2": [
                Candidate(segment=segments["s2"], score=0.90),
                Candidate(segment=segments["s1"], score=0.70),
            ],
            "q3": [
                Candidate(segment=segments["s3"], score=0.60),
            ],
        }
        mock_generator_cls.return_value = instance

        result = sample_hard_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=1,
            embedding_model_name="dummy/model",
        )
        by_query = {e.query_id: e.source_id for e in result}
        assert by_query["q1"] == "s2"
        assert by_query["q2"] == "s2"
        assert by_query["q3"] == "s3"
        assert all(e.label == "no_match" for e in result)

    @patch("locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator")
    def test_disables_prompts_for_a_generic_pretrained_model(
        self, mock_generator_cls, query_document, source_document, positives
    ):
        instance = MagicMock()
        instance.generate.return_value = {seg.id: [] for seg in query_document}
        mock_generator_cls.return_value = instance

        sample_hard_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=1,
            embedding_model_name="dummy/model",
        )

        _, call_kwargs = instance.generate.call_args
        assert call_kwargs["query_prompt_name"] == ""
        assert call_kwargs["source_prompt_name"] == ""

    @patch("locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator")
    def test_zero_n_per_query_skips_the_model_entirely(
        self, mock_generator_cls, query_document, source_document, positives
    ):
        result = sample_hard_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=0,
            embedding_model_name="dummy/model",
        )
        assert len(result) == 0
        mock_generator_cls.assert_not_called()

    @patch("locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator")
    def test_respects_n_per_query_cap(
        self, mock_generator_cls, query_document, source_document, positives
    ):
        segments = {seg.id: seg for seg in source_document}
        instance = MagicMock()
        instance.generate.return_value = {
            "q1": [Candidate(segment=segments[s], score=1.0) for s in ["s2", "s3", "s4", "s5"]],
            "q2": [],
            "q3": [],
        }
        mock_generator_cls.return_value = instance

        result = sample_hard_negatives(
            query_doc=query_document,
            source_doc=source_document,
            positives=positives,
            n_per_query=2,
            embedding_model_name="dummy/model",
        )
        q1_entries = [e for e in result if e.query_id == "q1"]
        assert len(q1_entries) == 2
        assert [e.source_id for e in q1_entries] == ["s2", "s3"]
