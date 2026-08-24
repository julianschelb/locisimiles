# pipeline/generator/tfidf.py
"""TF-IDF lexical candidate generator.

Ports the TF-IDF retrieval baseline used to evaluate the benchmark: CLTK
tokenization (optionally lemmatized) plus a ``scikit-learn`` TF-IDF vectorizer
fit on the source corpus, scored by cosine similarity (a plain sparse dot
product, since TF-IDF rows are L2-normalized).
"""

from __future__ import annotations

from typing import Any

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.generator._latin_text import NgramAnalyzer, preprocess


class TfidfCandidateGenerator(CandidateGeneratorBase):
    """Generate candidates by TF-IDF cosine similarity over Latin text.

    The vectorizer is fit on the source corpus only and queries are
    transformed with the same vocabulary, matching classic lexical retrieval
    (query-only terms are dropped rather than expanding the vocabulary).

    Args:
        lemmatize: Whether to lemmatize tokens with CLTK before vectorizing.
        lowercase: Whether to lowercase tokens before vectorizing.
        ngram_range: ``(min_n, max_n)`` token n-gram range, e.g. ``(1, 1)``
            for unigrams or ``(1, 2)`` for unigrams+bigrams.
        max_features: Maximum vocabulary size passed to ``TfidfVectorizer``.
        min_df: Minimum document frequency passed to ``TfidfVectorizer``.
        max_df: Maximum document frequency passed to ``TfidfVectorizer``.
        sublinear_tf: Whether to apply sublinear (``1 + log(tf)``) scaling.
    """

    def __init__(
        self,
        *,
        lemmatize: bool = True,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int = 50_000,
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
    ):
        self.lemmatize = bool(lemmatize)
        self.lowercase = bool(lowercase)
        self.ngram_range = ngram_range
        self.max_features = int(max_features)
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = bool(sublinear_tf)

        self._vectorizer: Any = None
        self._corpus_matrix: Any = None
        self._corpus_segment_ids: list[str] | None = None
        self._fitted_source_doc_id: int | None = None

    def _preprocess(self, text: str) -> list[str]:
        """Tokenize (and optionally lemmatize) one segment using this generator's settings."""
        return preprocess(text, lemmatize=self.lemmatize, lowercase=self.lowercase)

    def _fit_source(self, source_segments: list[TextSegment]) -> None:
        """Fit (or reuse a cached fit of) the TF-IDF vectorizer on the source corpus."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "TF-IDF support requires scikit-learn. "
                "Install it with: pip install 'locisimiles[lexical]'"
            ) from exc

        corpus_tokens = [self._preprocess(segment.text) for segment in source_segments]
        vectorizer = TfidfVectorizer(
            analyzer=NgramAnalyzer(self.ngram_range),
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
        )
        self._corpus_matrix = vectorizer.fit_transform(corpus_tokens)
        self._vectorizer = vectorizer
        self._corpus_segment_ids = [str(segment.id) for segment in source_segments]

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 100,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Generate top-k TF-IDF candidates for each query segment."""
        from sklearn.preprocessing import normalize

        eff_top_k = max(1, int(top_k))
        source_segments = list(source.segments.values())
        source_by_id = {str(segment.id): segment for segment in source_segments}

        if self._fitted_source_doc_id != id(source) or self._vectorizer is None:
            self._fit_source(source_segments)
            self._fitted_source_doc_id = id(source)

        assert self._corpus_segment_ids is not None
        query_segments = list(query.segments.values())
        query_tokens = [self._preprocess(segment.text) for segment in query_segments]
        query_matrix = self._vectorizer.transform(query_tokens)
        # Already L2-normalized via norm="l2", but out-of-vocabulary queries
        # can yield zero rows; normalize defensively as the reference does.
        query_matrix = normalize(query_matrix, norm="l2", copy=False)

        # Dense per-row similarities so zero-scoring source segments are still
        # candidates (matching BM25CandidateGenerator/Word2VecCandidateGenerator,
        # which always return up to top_k candidates rather than only nonzero
        # matches). Each row is small (n_source_segments floats); computing one
        # row at a time avoids densifying the whole (n_query, n_source) matrix.
        similarities = query_matrix @ self._corpus_matrix.T

        results: CandidateGeneratorOutput = {}
        for row_idx, query_segment in enumerate(query_segments):
            row = similarities.getrow(row_idx).toarray().ravel()
            scored = [
                Candidate(segment=source_by_id[seg_id], score=float(score))
                for seg_id, score in zip(self._corpus_segment_ids, row)
            ]
            scored.sort(key=lambda item: item.score, reverse=True)
            results[str(query_segment.id)] = scored[:eff_top_k]

        return results
