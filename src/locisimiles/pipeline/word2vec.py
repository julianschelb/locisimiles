"""Word2Vec retrieval pipeline using threshold-based judging."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from locisimiles.pipeline.generator.word2vec import (
    DEFAULT_WORD2VEC_MODEL_PATH,
    Word2VecCandidateGenerator,
)
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.pipeline import Pipeline


class Word2VecRetrievalPipeline(Pipeline):
    """Burns-style Word2Vec retrieval pipeline.

    Args:
        model_path: Path to a local gensim ``.model`` file.
        top_k: Number of candidates to mark as positive via threshold judge.
        similarity_threshold: Optional score threshold for positive labels.
        interval: Maximum token gap for bigrams.
        order_free: Whether to treat bigrams as order-insensitive.
    """

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_WORD2VEC_MODEL_PATH,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        interval: int = 0,
        order_free: bool = False,
    ):
        super().__init__(
            generator=Word2VecCandidateGenerator(
                model_path=model_path,
                interval=interval,
                order_free=order_free,
            ),
            judge=ThresholdJudge(top_k=top_k, similarity_threshold=similarity_threshold),
        )
