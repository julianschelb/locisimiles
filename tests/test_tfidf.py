"""Tests for the TF-IDF candidate generator and retrieval pipeline."""

from __future__ import annotations

from locisimiles.document import Document
from locisimiles.pipeline._types import Candidate


class TestTfidfCandidateGenerator:
    def test_generate_returns_ranked_candidates_for_all_source_segments(
        self, query_document, source_document
    ):
        """Generator returns up to top_k candidates for every source segment,
        including zero-similarity ones (consistent with BM25/Word2Vec)."""
        from locisimiles.pipeline.generator.tfidf import TfidfCandidateGenerator

        generator = TfidfCandidateGenerator()
        result = generator.generate(query=query_document, source=source_document, top_k=5)

        assert isinstance(result, dict)
        assert set(result.keys()) == {seg.id for seg in query_document}
        for _qid, candidates in result.items():
            assert len(candidates) <= 5
            assert all(isinstance(c, Candidate) for c in candidates)
            scores = [c.score for c in candidates]
            assert scores == sorted(scores, reverse=True)

    def test_ranks_near_duplicate_source_segment_first(self, temp_dir):
        """A near-duplicate source segment should rank above unrelated ones."""
        from locisimiles.pipeline.generator.tfidf import TfidfCandidateGenerator

        query_path = temp_dir / "query.csv"
        query_path.write_text(
            "seg_id,text\nq1,Arma virumque cano Troiae qui primus ab oris.\n",
            encoding="utf-8",
        )
        source_path = temp_dir / "source.csv"
        source_path.write_text(
            "seg_id,text\n"
            "s1,Arma virumque cano qui primus Troiae.\n"
            "s2,Nothing at all similar in vocabulary here.\n",
            encoding="utf-8",
        )

        generator = TfidfCandidateGenerator()
        result = generator.generate(
            query=Document(query_path), source=Document(source_path), top_k=2
        )
        top = result["q1"][0]
        assert top.segment.id == "s1"
        assert top.score > result["q1"][1].score

    def test_respects_top_k(self, query_document, source_document):
        from locisimiles.pipeline.generator.tfidf import TfidfCandidateGenerator

        generator = TfidfCandidateGenerator()
        result = generator.generate(query=query_document, source=source_document, top_k=2)
        for candidates in result.values():
            assert len(candidates) <= 2


class TestTfidfRetrievalPipeline:
    def test_pipeline_composition(self, query_document, source_document):
        from locisimiles.pipeline.judge.threshold import ThresholdJudge
        from locisimiles.pipeline.tfidf import TfidfRetrievalPipeline

        pipeline = TfidfRetrievalPipeline(top_k=3)
        assert isinstance(pipeline.judge, ThresholdJudge)
        assert pipeline.judge.top_k == 3
