"""Contextual retrieval + classification two-stage pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from locisimiles.pipeline.generator.contextual_bert import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    LatinBertContextualCandidateGenerator,
)
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.pipeline import Pipeline


class LatinBertTwoStagePipeline(Pipeline):
    """Two-stage pipeline with contextual Latin BERT retrieval.

    Stage 1 performs contextual token-level retrieval.
    Stage 2 applies a sequence classifier for reranking/labeling.
    """

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
        model_name: str = DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
        model_path: str | Path | None = None,
        device: str | int | None = None,
        pos_class_idx: int = 1,
        label_names: Sequence[str] | Mapping[int | str, str] | None = None,
        positive_class_ids: Sequence[int] | None = None,
        positive_labels: Sequence[str] | None = None,
        negative_labels: Sequence[str] | None = None,
        emit_class_metadata: bool | None = None,
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
            judge=ClassificationJudge(
                classification_name=classification_name,
                device=device,
                pos_class_idx=pos_class_idx,
                label_names=label_names,
                positive_class_ids=positive_class_ids,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                emit_class_metadata=emit_class_metadata,
            ),
        )
