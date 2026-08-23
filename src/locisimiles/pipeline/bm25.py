# pipeline/bm25.py
"""BM25 retrieval pipeline using threshold-based judging."""

from __future__ import annotations

from typing import Optional

from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.pipeline import Pipeline


class BM25RetrievalPipeline(Pipeline):
    """BM25 (Okapi) lexical retrieval pipeline — the benchmark's best retriever.

    Args:
        top_k: Number of candidates to mark as positive via threshold judge.
        similarity_threshold: Optional score threshold for positive labels.
        lemmatize: Whether to lemmatize tokens with CLTK before indexing.
        lowercase: Whether to lowercase tokens before indexing.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        lemmatize: bool = True,
        lowercase: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        super().__init__(
            generator=BM25CandidateGenerator(
                lemmatize=lemmatize,
                lowercase=lowercase,
                k1=k1,
                b=b,
            ),
            judge=ThresholdJudge(top_k=top_k, similarity_threshold=similarity_threshold),
        )
