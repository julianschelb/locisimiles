# pipeline/generator/bm25.py
"""BM25 lexical candidate generator.

Ports the BM25 (Okapi) retrieval baseline used to evaluate the benchmark —
the strongest retriever reported in the paper. Uses CLTK tokenization
(optionally lemmatized) and ``rank_bm25.BM25Okapi`` with the standard
Okapi hyperparameters (``k1=1.5``, ``b=0.75``).
"""

from __future__ import annotations

from typing import Any

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.generator._latin_text import preprocess


class BM25CandidateGenerator(CandidateGeneratorBase):
    """Generate candidates by Okapi BM25 score over Latin text.

    Args:
        lemmatize: Whether to lemmatize tokens with CLTK before indexing.
        lowercase: Whether to lowercase tokens before indexing.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.
    """

    def __init__(
        self,
        *,
        lemmatize: bool = True,
        lowercase: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.lemmatize = bool(lemmatize)
        self.lowercase = bool(lowercase)
        self.k1 = float(k1)
        self.b = float(b)

        self._index: Any = None
        self._corpus_segment_ids: list[str] | None = None
        self._fitted_source_doc_id: int | None = None

    def _preprocess(self, text: str) -> list[str]:
        return preprocess(text, lemmatize=self.lemmatize, lowercase=self.lowercase)

    def _fit_source(self, source_segments: list[TextSegment]) -> None:
        """Build (or reuse a cached build of) the BM25 index over the source corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25 support requires rank_bm25. "
                "Install it with: pip install 'locisimiles[lexical]'"
            ) from exc

        corpus_tokens = [self._preprocess(segment.text) for segment in source_segments]
        self._index = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
        self._corpus_segment_ids = [str(segment.id) for segment in source_segments]

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 100,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Generate top-k BM25 candidates for each query segment."""
        eff_top_k = max(1, int(top_k))
        source_segments = list(source.segments.values())
        source_by_id = {str(segment.id): segment for segment in source_segments}

        if self._fitted_source_doc_id != id(source) or self._index is None:
            self._fit_source(source_segments)
            self._fitted_source_doc_id = id(source)

        assert self._corpus_segment_ids is not None
        results: CandidateGeneratorOutput = {}
        for query_segment in query.segments.values():
            query_tokens = self._preprocess(query_segment.text)
            scores = self._index.get_scores(query_tokens)
            scored = [
                Candidate(segment=source_by_id[seg_id], score=float(score))
                for seg_id, score in zip(self._corpus_segment_ids, scores)
            ]
            scored.sort(key=lambda item: item.score, reverse=True)
            results[str(query_segment.id)] = scored[:eff_top_k]

        return results
