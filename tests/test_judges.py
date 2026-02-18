"""
Unit tests for locisimiles.pipeline.judge subpackage.

Tests the modular judge components:
- IdentityJudge     (no ML dependencies)
- ThresholdJudge    (no ML dependencies)
- ClassificationJudge (requires mocking)
- JudgeBase         (ABC contract)
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import (
    Candidate,
    CandidateJudge,
    CandidateGeneratorOutput,
    CandidateJudgeOutput,
)
from locisimiles.pipeline.judge._base import JudgeBase


# ============== Fixtures ==============


@pytest.fixture
def sample_candidates(sample_segments):
    """Create CandidateGeneratorOutput for judge testing."""
    return {
        "q1": [
            Candidate(segment=sample_segments[0], score=0.95),
            Candidate(segment=sample_segments[1], score=0.75),
            Candidate(segment=sample_segments[2], score=0.55),
        ],
        "q2": [
            Candidate(segment=sample_segments[1], score=0.88),
            Candidate(segment=sample_segments[0], score=0.35),
        ],
    }


@pytest.fixture
def judge_query_document(temp_dir):
    """Create a query document whose IDs match sample_candidates keys."""
    csv_path = temp_dir / "judge_query.csv"
    csv_path.write_text(
        "seg_id,text\n"
        "q1,First query segment text.\n"
        "q2,Second query segment text.\n",
        encoding="utf-8",
    )
    return Document(csv_path)


# ============== ABC Contract ==============


class TestJudgeBase:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abc(self):
        """JudgeBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            JudgeBase()

    def test_subclass_must_implement_judge(self):
        """Subclasses that don't implement judge() cannot be instantiated."""

        class IncompleteJudge(JudgeBase):
            pass

        with pytest.raises(TypeError):
            IncompleteJudge()

    def test_subclass_with_judge_works(self):
        """A proper subclass can be instantiated."""

        class DummyJudge(JudgeBase):
            def judge(self, *, query, candidates, **kwargs):
                return {}

        j = DummyJudge()
        assert isinstance(j, JudgeBase)


# ============== IdentityJudge ==============


class TestIdentityJudge:
    """Tests for the identity (pass-through) judge."""

    def test_judgment_score_always_one(self, judge_query_document, sample_candidates):
        """Every candidate should get judgment_score=1.0."""
        from locisimiles.pipeline.judge.identity import IdentityJudge

        judge = IdentityJudge()
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        assert isinstance(result, dict)
        for qid, judgments in result.items():
            for j in judgments:
                assert isinstance(j, CandidateJudge)
                assert j.judgment_score == 1.0

    def test_preserves_candidate_score(self, judge_query_document, sample_candidates):
        """candidate_score should be preserved from the input."""
        from locisimiles.pipeline.judge.identity import IdentityJudge

        judge = IdentityJudge()
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        for qid in sample_candidates:
            for orig, judged in zip(sample_candidates[qid], result[qid]):
                assert judged.candidate_score == orig.score

    def test_preserves_segments(self, judge_query_document, sample_candidates):
        """Segments should pass through unchanged."""
        from locisimiles.pipeline.judge.identity import IdentityJudge

        judge = IdentityJudge()
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        for qid in sample_candidates:
            for orig, judged in zip(sample_candidates[qid], result[qid]):
                assert judged.segment is orig.segment

    def test_preserves_query_ids(self, judge_query_document, sample_candidates):
        """Output keys should match input keys."""
        from locisimiles.pipeline.judge.identity import IdentityJudge

        judge = IdentityJudge()
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)
        assert set(result.keys()) == set(sample_candidates.keys())

    def test_is_judge_base(self):
        """IdentityJudge is a JudgeBase."""
        from locisimiles.pipeline.judge.identity import IdentityJudge

        assert issubclass(IdentityJudge, JudgeBase)


# ============== ThresholdJudge ==============


class TestThresholdJudge:
    """Tests for the threshold-based judge."""

    def test_top_k_default(self, judge_query_document, sample_candidates):
        """Default top_k=10 should mark all candidates as positive when fewer exist."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        judge = ThresholdJudge(top_k=10)
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        for qid, judgments in result.items():
            for j in judgments:
                assert j.judgment_score == 1.0

    def test_top_k_limits(self, judge_query_document, sample_candidates):
        """Only the first top_k candidates should get judgment_score=1.0."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        judge = ThresholdJudge(top_k=1)
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        for qid, judgments in result.items():
            assert judgments[0].judgment_score == 1.0
            for j in judgments[1:]:
                assert j.judgment_score == 0.0

    def test_top_k_override_at_judge_time(self, judge_query_document, sample_candidates):
        """top_k can be overridden in the judge() call."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        judge = ThresholdJudge(top_k=10)
        result = judge.judge(
            query=judge_query_document,
            candidates=sample_candidates,
            top_k=1,
        )

        for qid, judgments in result.items():
            positive = [j for j in judgments if j.judgment_score == 1.0]
            assert len(positive) == 1

    def test_similarity_threshold(self, judge_query_document, sample_candidates):
        """Candidates above the threshold should be positive."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        judge = ThresholdJudge(similarity_threshold=0.80)
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        for qid, judgments in result.items():
            for j in judgments:
                if j.candidate_score >= 0.80:
                    assert j.judgment_score == 1.0
                else:
                    assert j.judgment_score == 0.0

    def test_similarity_threshold_override(self, judge_query_document, sample_candidates):
        """similarity_threshold can be overridden in the judge() call."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        judge = ThresholdJudge(similarity_threshold=0.99)  # Very high
        result = judge.judge(
            query=judge_query_document,
            candidates=sample_candidates,
            similarity_threshold=0.50,  # Override to lower
        )

        # With threshold=0.50, candidates with score >= 0.50 should be positive
        for qid, judgments in result.items():
            for j in judgments:
                if j.candidate_score >= 0.50:
                    assert j.judgment_score == 1.0

    def test_preserves_candidate_score(self, judge_query_document, sample_candidates):
        """candidate_score should be preserved."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        judge = ThresholdJudge(top_k=2)
        result = judge.judge(query=judge_query_document, candidates=sample_candidates)

        for qid in sample_candidates:
            for orig, judged in zip(sample_candidates[qid], result[qid]):
                assert judged.candidate_score == orig.score

    def test_is_judge_base(self):
        """ThresholdJudge is a JudgeBase."""
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        assert issubclass(ThresholdJudge, JudgeBase)


# ============== ClassificationJudge ==============


class TestClassificationJudge:
    """Tests for the classification judge (mocked models)."""

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test that the judge initializes with classifier model and tokenizer."""
        from locisimiles.pipeline.judge.classification import ClassificationJudge

        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        judge = ClassificationJudge(device="cpu")

        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        assert judge.device == "cpu"

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_truncate_pair(self, mock_tokenizer_class, mock_model_class):
        """Test text pair truncation."""
        from locisimiles.pipeline.judge.classification import ClassificationJudge

        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()[:250]
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        judge = ClassificationJudge(device="cpu")

        long_text = " ".join(["word"] * 300)
        trunc1, trunc2 = judge._truncate_pair(long_text, long_text)

        assert len(trunc1.split()) <= 255
        assert len(trunc2.split()) <= 255

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    def test_predict_batch(self, mock_tokenizer_class, mock_model_class):
        """Test batch prediction returns correct number of probabilities."""
        from locisimiles.pipeline.judge.classification import ClassificationJudge

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

        judge = ClassificationJudge(device="cpu")
        probs = judge._predict_batch("query text", ["c1", "c2", "c3"])

        assert len(probs) == 3
        for p in probs:
            assert 0.0 <= p <= 1.0

    @patch("locisimiles.pipeline.judge.classification.AutoModelForSequenceClassification")
    @patch("locisimiles.pipeline.judge.classification.AutoTokenizer")
    @patch("locisimiles.pipeline.judge.classification.tqdm", lambda x, **kwargs: x)
    def test_judge(self, mock_tokenizer_class, mock_model_class, temp_dir):
        """Test judge() returns CandidateJudgeOutput."""
        from locisimiles.pipeline.judge.classification import ClassificationJudge

        # Setup tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.num_special_tokens_to_add.return_value = 3
        mock_tokenizer.tokenize.side_effect = lambda x: x.split()
        mock_tokenizer.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Setup model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_result = MagicMock()
        mock_result.logits = torch.randn(2, 2)
        mock_model.return_value = mock_result
        mock_model_class.from_pretrained.return_value = mock_model

        judge = ClassificationJudge(device="cpu")

        # Create query document
        query_csv = temp_dir / "query.csv"
        query_csv.write_text(
            "seg_id,text\nq1,Query text one.\nq2,Query text two.\n",
            encoding="utf-8",
        )
        query_doc = Document(query_csv)

        # Create candidates
        source_segs = [
            TextSegment("Source one.", "s1", row_id=0),
            TextSegment("Source two.", "s2", row_id=1),
        ]
        candidates: CandidateGeneratorOutput = {
            "q1": [Candidate(segment=source_segs[0], score=0.9),
                   Candidate(segment=source_segs[1], score=0.7)],
            "q2": [Candidate(segment=source_segs[0], score=0.8),
                   Candidate(segment=source_segs[1], score=0.6)],
        }

        result = judge.judge(query=query_doc, candidates=candidates)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"q1", "q2"}
        for qid, judgments in result.items():
            assert len(judgments) == 2
            for j in judgments:
                assert isinstance(j, CandidateJudge)
                assert 0.0 <= j.judgment_score <= 1.0
                assert j.candidate_score is not None

    def test_is_judge_base(self):
        """ClassificationJudge is a JudgeBase."""
        from locisimiles.pipeline.judge.classification import ClassificationJudge

        assert issubclass(ClassificationJudge, JudgeBase)


# ============== Import Tests ==============


class TestJudgeImports:
    """Test that judges are importable from the expected paths."""

    def test_import_from_judge_package(self):
        """All judges should be importable from the judge package."""
        from locisimiles.pipeline.judge import (
            JudgeBase,
            ClassificationJudge,
            ThresholdJudge,
            IdentityJudge,
        )

    def test_import_from_pipeline_package(self):
        """All judges should be importable from the pipeline package."""
        from locisimiles.pipeline import (
            JudgeBase,
            ClassificationJudge,
            ThresholdJudge,
            IdentityJudge,
        )

    def test_import_from_top_level(self):
        """Key judges should be importable from the top-level package."""
        from locisimiles import (
            ClassificationJudge,
            ThresholdJudge,
            IdentityJudge,
        )
