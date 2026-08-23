# pipeline/tfidf.py
"""TF-IDF retrieval pipeline using threshold-based judging."""

from __future__ import annotations

from typing import Optional

from locisimiles.pipeline.generator.tfidf import TfidfCandidateGenerator
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.pipeline import Pipeline


class TfidfRetrievalPipeline(Pipeline):
    """TF-IDF lexical retrieval pipeline.

    Args:
        top_k: Number of candidates to mark as positive via threshold judge.
        similarity_threshold: Optional score threshold for positive labels.
        lemmatize: Whether to lemmatize tokens with CLTK before vectorizing.
        lowercase: Whether to lowercase tokens before vectorizing.
        ngram_range: ``(min_n, max_n)`` token n-gram range.
        max_features: Maximum vocabulary size.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        lemmatize: bool = True,
        lowercase: bool = True,
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int = 50_000,
    ):
        super().__init__(
            generator=TfidfCandidateGenerator(
                lemmatize=lemmatize,
                lowercase=lowercase,
                ngram_range=ngram_range,
                max_features=max_features,
            ),
            judge=ThresholdJudge(top_k=top_k, similarity_threshold=similarity_threshold),
        )
