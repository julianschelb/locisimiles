"""Contextual Latin-BERT retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from locisimiles.pipeline.generator.contextual_bert import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    LatinBertContextualCandidateGenerator,
)
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.pipeline import Pipeline


class LatinBertRetrievalPipeline(Pipeline):
    """Retrieval-only pipeline using contextual token similarity.

    Args:
        model_name: HuggingFace model identifier.
        model_path: Optional local path to a model directory.
        device: Torch device.
        top_k: Number of candidates marked positive by default.
        similarity_threshold: Optional score threshold overriding top-k labels.
        max_length: Maximum tokenized length per segment.
        min_token_length: Minimum token length retained for scoring.
        use_stopword_filter: Whether to filter common Latin stopwords.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
        model_path: str | Path | None = None,
        device: str | int | None = None,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        max_length: int = 256,
        min_token_length: int = 2,
        use_stopword_filter: bool = True,
    ):
        super().__init__(
            generator=LatinBertContextualCandidateGenerator(
                model_name=model_name,
                model_path=model_path,
                device=device,
                max_length=max_length,
                min_token_length=min_token_length,
                use_stopword_filter=use_stopword_filter,
            ),
            judge=ThresholdJudge(top_k=top_k, similarity_threshold=similarity_threshold),
        )
