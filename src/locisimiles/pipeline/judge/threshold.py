# pipeline/judge/threshold.py
"""Threshold judge — binary decisions based on candidate scores."""
from __future__ import annotations

from typing import Any, Optional

from locisimiles.document import Document
from locisimiles.pipeline._types import (
    CandidateJudge,
    CandidateGeneratorOutput,
    CandidateJudgeOutput,
)
from locisimiles.pipeline.judge._base import JudgeBase


class ThresholdJudge(JudgeBase):
    """Judge candidates using a simple score threshold or top-k cut-off.

    Two strategies are available (mutually exclusive):

    * **Top-k** (default): the first *top_k* candidates per query (assumed
      to be sorted by score descending) receive ``judgment_score = 1.0``;
      the rest get ``0.0``.
    * **Similarity threshold**: if *similarity_threshold* is provided,
      every candidate whose ``score >= similarity_threshold`` receives
      ``judgment_score = 1.0``.

    Args:
        top_k: Number of top candidates to mark as positive.
        similarity_threshold: Score threshold for positive decisions.

    Example:
        ```python
        from locisimiles.pipeline.judge import ThresholdJudge

        judge = ThresholdJudge(top_k=5)
        results = judge.judge(query=query_doc, candidates=candidates)
        ```
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def judge(
        self,
        *,
        query: Document,
        candidates: CandidateGeneratorOutput,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Apply threshold or top-k rule to produce binary judgments.

        Args:
            query: Query document (unused but required by protocol).
            candidates: Output from a candidate generator.
            top_k: Override instance ``top_k``.
            similarity_threshold: Override instance ``similarity_threshold``.

        Returns:
            ``CandidateJudgeOutput`` with ``judgment_score`` ∈ {0.0, 1.0}.
        """
        _top_k = top_k if top_k is not None else self.top_k
        _threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        judge_results: CandidateJudgeOutput = {}

        for query_id, candidate_list in candidates.items():
            judge_results[query_id] = []

            for rank, candidate in enumerate(candidate_list):
                if _threshold is not None:
                    is_positive = candidate.score >= _threshold
                else:
                    is_positive = rank < _top_k

                judge_results[query_id].append(
                    CandidateJudge(
                        segment=candidate.segment,
                        candidate_score=candidate.score,
                        judgment_score=1.0 if is_positive else 0.0,
                    )
                )

        return judge_results
