# pipeline/bm25_lexical_two_stage.py
"""BM25 retrieval + lexical classification two-stage pipeline.

Combines BM25 retrieval with a trained LogReg/GBDT lexical classifier — the
"best non-neural" configuration reported for the benchmark.
"""

from __future__ import annotations

from typing import Sequence

from locisimiles.pipeline.generator.bm25 import BM25CandidateGenerator
from locisimiles.pipeline.judge.lexical_classifier import LexicalClassifierJudge
from locisimiles.pipeline.pipeline import Pipeline


class BM25LexicalTwoStagePipeline(Pipeline):
    """Two-stage pipeline with BM25 retrieval and a lexical classifier reranker.

    Stage 1 ranks candidates by Okapi BM25 score over (optionally
    lemmatized) Latin text. Stage 2 applies a trained LogReg/GBDT classifier
    (TF-IDF/Jaccard/overlap features) for reranking/labeling — no neural
    model required end to end.
    """

    def __init__(
        self,
        *,
        artifact_path: str,
        pos_class_idx: int = 1,
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
            judge=LexicalClassifierJudge(
                artifact_path=artifact_path,
                pos_class_idx=pos_class_idx,
                positive_class_ids=positive_class_ids,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                emit_class_metadata=emit_class_metadata,
            ),
        )
