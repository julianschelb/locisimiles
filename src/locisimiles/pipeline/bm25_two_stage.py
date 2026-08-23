# pipeline/bm25_two_stage.py
"""BM25 retrieval + classification two-stage pipeline.

Combines the benchmark's strongest retriever (BM25) with a fine-tuned
cross-encoder classifier — the "best combined" configuration reported for
the benchmark.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.pipeline import Pipeline


class BM25TwoStagePipeline(Pipeline):
    """Two-stage pipeline with BM25 retrieval.

    Stage 1 ranks candidates by Okapi BM25 score over (optionally
    lemmatized) Latin text. Stage 2 applies a sequence classifier for
    reranking/labeling.
    """

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,
        label_names: Sequence[str] | Mapping[int | str, str] | None = None,
        positive_class_ids: Sequence[int] | None = None,
        positive_labels: Sequence[str] | None = None,
        negative_labels: Sequence[str] | None = None,
        emit_class_metadata: bool | None = None,
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
