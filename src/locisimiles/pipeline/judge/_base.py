# pipeline/judge/_base.py
"""Abstract base class for judges."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from locisimiles.document import Document
from locisimiles.pipeline._types import CandidateGeneratorOutput, CandidateJudgeOutput


class JudgeBase(ABC):
    """Abstract base class for candidate judges.

    A judge receives the output of a candidate generator and produces a
    ``CandidateJudgeOutput`` — a dictionary mapping query-segment IDs to
    lists of ``CandidateJudge`` objects, each containing a source segment,
    the original candidate score, and a final judgment score.

    Subclasses must implement ``judge()``.

    Available implementations:

    - ``ClassificationJudge`` — scores pairs with a fine-tuned
      transformer classification model.
    - ``ThresholdJudge`` — applies a top-k or score-threshold rule.
    - ``IdentityJudge`` — passes candidates through unchanged
      (``judgment_score = 1.0``).
    """

    @abstractmethod
    def judge(
        self,
        *,
        query: Document,
        candidates: CandidateGeneratorOutput,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Score or classify candidates.

        Args:
            query: Query document (needed to look up query-segment texts).
            candidates: Output from a candidate generator.
            **kwargs: Judge-specific parameters.

        Returns:
            Mapping of query segment IDs → lists of ``CandidateJudge`` objects.
        """
        ...
