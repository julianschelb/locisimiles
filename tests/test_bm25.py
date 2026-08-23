"""Tests for the BM25 candidate generator and retrieval pipeline."""

from __future__ import annotations

import pytest

from locisimiles.document import Document
from locisimiles.pipeline._types import Candidate

pytest.importorskip("cltk", reason="cltk has no release supporting this Python version")


class TestBM25CandidateGenerator:
    def test_generate_returns_ranked_candidates_for_all_source_segments(
        self, query_document, source_document
    ):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator

        generator = BM25CandidateGenerator()
        result = generator.generate(query=query_document, source=source_document, top_k=5)

        assert isinstance(result, dict)
        assert set(result.keys()) == {seg.id for seg in query_document}
        for _qid, candidates in result.items():
            assert len(candidates) <= 5
            assert all(isinstance(c, Candidate) for c in candidates)
            scores = [c.score for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_ranks_near_duplicate_source_segment_first(self, temp_dir):
        """A near-duplicate source segment should score highest and positive."""
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator

        query_path = temp_dir / "query.csv"
        query_path.write_text(
            "seg_id,text\nq1,Arma virumque cano Troiae qui primus ab oris.\n",
            encoding="utf-8",
        )
        source_path = temp_dir / "source.csv"
        source_path.write_text(
            "seg_id,text\n"
            "s1,Arma virumque cano qui primus Troiae.\n"
            "s2,Fato profugus Italiam venit.\n"
            "s3,Multum terris iactatus et alto litora.\n"
            "s4,Completely unrelated text about cooking recipes.\n"
            "s5,Another unrelated segment discussing modern politics.\n",
            encoding="utf-8",
        )

        generator = BM25CandidateGenerator()
        result = generator.generate(
            query=Document(query_path), source=Document(source_path), top_k=5
        )
        top = result["q1"][0]
        assert top.segment.id == "s1"
        assert top.score > 0
        assert top.score > result["q1"][1].score

    def test_respects_top_k(self, query_document, source_document):
        from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator

        generator = BM25CandidateGenerator()
        result = generator.generate(query=query_document, source=source_document, top_k=2)
        for candidates in result.values():
            assert len(candidates) <= 2


class TestBM25RetrievalPipeline:
    def test_pipeline_composition(self):
        from locisimiles.pipeline.bm25 import BM25RetrievalPipeline
        from locisimiles.pipeline.judge.threshold import ThresholdJudge

        pipeline = BM25RetrievalPipeline(top_k=4)
        assert isinstance(pipeline.judge, ThresholdJudge)
        assert pipeline.judge.top_k == 4
