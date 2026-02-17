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
    ``CandidateJudgeOutput`` mapping query-segment IDs to lists of
    ``CandidateJudge`` objects.

    Subclasses must implement ``judge()``.
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
