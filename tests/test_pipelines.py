"""
Unit tests for locisimiles.pipeline.pipeline module (Pipeline composer).

Tests the generic Pipeline(generator, judge) composition.
"""

from unittest.mock import MagicMock

import pytest

from locisimiles.pipeline._types import (
    Candidate,
    CandidateJudge,
)
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.judge._base import JudgeBase
from locisimiles.pipeline.pipeline import Pipeline

# ============== Fixtures ==============


@pytest.fixture
def mock_generator(sample_segments):
    """Create a mock generator that returns fixed candidates."""
    candidates = {
        "q1": [
            Candidate(segment=sample_segments[0], score=0.9),
            Candidate(segment=sample_segments[1], score=0.7),
        ],
        "q2": [
            Candidate(segment=sample_segments[2], score=0.5),
        ],
    }

    gen = MagicMock(spec=CandidateGeneratorBase)
    gen.generate.return_value = candidates
    return gen, candidates


@pytest.fixture
def mock_judge(sample_segments):
    """Create a mock judge that returns fixed judgments."""
    judgments = {
        "q1": [
            CandidateJudge(segment=sample_segments[0], candidate_score=0.9, judgment_score=0.85),
            CandidateJudge(segment=sample_segments[1], candidate_score=0.7, judgment_score=0.45),
        ],
        "q2": [
            CandidateJudge(segment=sample_segments[2], candidate_score=0.5, judgment_score=0.25),
        ],
    }

    j = MagicMock(spec=JudgeBase)
    j.judge.return_value = judgments
    return j, judgments


# ============== Pipeline Composition ==============


class TestPipelineComposition:
    """Tests for Pipeline(generator, judge) composition."""

    def test_run_calls_both_stages(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """run() should call generator.generate() then judge.judge()."""
        gen, _ = mock_generator
        judge, _ = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        result = pipeline.run(query=query_document, source=source_document)

        gen.generate.assert_called_once()
        judge.judge.assert_called_once()
        assert isinstance(result, dict)

    def test_run_returns_judge_output(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """run() should return the judge's output."""
        gen, _ = mock_generator
        judge, expected = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        result = pipeline.run(query=query_document, source=source_document)

        assert result == expected

    def test_generate_candidates_only(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """generate_candidates() should only call the generator."""
        gen, expected_candidates = mock_generator
        judge, _ = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        result = pipeline.generate_candidates(query=query_document, source=source_document)

        gen.generate.assert_called_once()
        judge.judge.assert_not_called()
        assert result == expected_candidates

    def test_judge_candidates_only(self, mock_generator, mock_judge, query_document):
        """judge_candidates() should only call the judge."""
        gen, candidates = mock_generator
        judge, expected_judgments = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        result = pipeline.judge_candidates(query=query_document, candidates=candidates)

        gen.generate.assert_not_called()
        judge.judge.assert_called_once()
        assert result == expected_judgments

    def test_kwargs_forwarded_to_generator(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """Extra kwargs should be forwarded to generator.generate()."""
        gen, _ = mock_generator
        judge, _ = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        pipeline.run(
            query=query_document,
            source=source_document,
            top_k=5,
            query_prompt_name="test",
        )

        call_kwargs = gen.generate.call_args[1]
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["query_prompt_name"] == "test"

    def test_kwargs_forwarded_to_judge(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """Extra kwargs should be forwarded to judge.judge()."""
        gen, _ = mock_generator
        judge, _ = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        pipeline.run(
            query=query_document,
            source=source_document,
            batch_size=16,
        )

        call_kwargs = judge.judge.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_caches_last_candidates(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """Pipeline should cache the last candidates."""
        gen, expected_candidates = mock_generator
        judge, _ = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        assert pipeline._last_candidates is None

        pipeline.run(query=query_document, source=source_document)

        assert pipeline._last_candidates == expected_candidates

    def test_caches_last_judgments(
        self, mock_generator, mock_judge, query_document, source_document
    ):
        """Pipeline should cache the last judgments."""
        gen, _ = mock_generator
        judge, expected_judgments = mock_judge

        pipeline = Pipeline(generator=gen, judge=judge)
        assert pipeline._last_judgments is None

        pipeline.run(query=query_document, source=source_document)

        assert pipeline._last_judgments == expected_judgments


# ============== Pipeline with Real Components ==============


class TestPipelineWithRealComponents:
    """Integration tests with actual (non-mocked) generator/judge components."""

    def test_exhaustive_plus_identity(self, query_document, source_document):
        """Pipeline(ExhaustiveCandidateGenerator, IdentityJudge) should work."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
        from locisimiles.pipeline.judge.identity import IdentityJudge

        pipeline = Pipeline(
            generator=ExhaustiveCandidateGenerator(),
            judge=IdentityJudge(),
        )
        result = pipeline.run(query=query_document, source=source_document)

        assert isinstance(result, dict)
        n_source = len(list(source_document))
        for qid, judgments in result.items():
            assert len(judgments) == n_source
            for j in judgments:
                assert isinstance(j, CandidateJudge)
                assert j.judgment_score == 1.0
                assert j.candidate_score == 1.0

    def test_exhaustive_plus_threshold(self, query_document, source_document):
        """Pipeline(ExhaustiveCandidateGenerator, ThresholdJudge) should work."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        pipeline = Pipeline(
            generator=ExhaustiveCandidateGenerator(),
            judge=ThresholdJudge(top_k=2),
        )
        result = pipeline.run(query=query_document, source=source_document)

        for qid, judgments in result.items():
            positive = [j for j in judgments if j.judgment_score == 1.0]
            negative = [j for j in judgments if j.judgment_score == 0.0]
            assert len(positive) == 2
            assert len(negative) == len(judgments) - 2

    def test_rule_based_plus_identity(self, query_document, source_document):
        """Pipeline(RuleBasedCandidateGenerator, IdentityJudge) should work."""
        from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator
        from locisimiles.pipeline.judge.identity import IdentityJudge

        pipeline = Pipeline(
            generator=RuleBasedCandidateGenerator(min_shared_words=2),
            judge=IdentityJudge(),
        )
        result = pipeline.run(
            query=query_document,
            source=source_document,
            query_genre="prose",
            source_genre="poetry",
        )

        assert isinstance(result, dict)
        for qid, judgments in result.items():
            for j in judgments:
                assert isinstance(j, CandidateJudge)
                assert j.judgment_score == 1.0

    def test_two_stage_generate_then_judge_separately(self, query_document, source_document):
        """Test using generate_candidates() then judge_candidates() separately."""
        from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        pipeline = Pipeline(
            generator=ExhaustiveCandidateGenerator(),
            judge=ThresholdJudge(top_k=1),
        )

        candidates = pipeline.generate_candidates(query=query_document, source=source_document)
        assert isinstance(candidates, dict)
        for qid, cands in candidates.items():
            assert all(isinstance(c, Candidate) for c in cands)

        result = pipeline.judge_candidates(query=query_document, candidates=candidates)
        assert isinstance(result, dict)
        for qid, judgments in result.items():
            positive = [j for j in judgments if j.judgment_score == 1.0]
            assert len(positive) == 1


# ============== Import Tests ==============


class TestPipelineImports:
    """Test that Pipeline is importable from the expected paths."""

    def test_import_from_pipeline_module(self):
        """Pipeline should be importable from the pipeline module."""

    def test_import_from_pipeline_package(self):
        """Pipeline should be importable from the pipeline package."""

    def test_import_from_top_level(self):
        """Pipeline should be importable from the top-level package."""
