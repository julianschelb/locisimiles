# pipeline/judge/identity.py
"""Identity judge — passes candidates through unchanged."""
from __future__ import annotations

from typing import Any

from locisimiles.document import Document
from locisimiles.pipeline._types import (
    CandidateJudge,
    CandidateGeneratorOutput,
    CandidateJudgeOutput,
)
from locisimiles.pipeline.judge._base import JudgeBase


class IdentityJudge(JudgeBase):
    """Pass every candidate through with ``judgment_score = 1.0``.

    Useful when the candidate generator already performs all the
    filtering and scoring that is needed (e.g. the rule-based generator).
    No additional models are loaded.

    Example:
        ```python
        from locisimiles.pipeline.judge import IdentityJudge

        judge = IdentityJudge()
        results = judge.judge(query=query_doc, candidates=candidates)

        # Every candidate gets judgment_score = 1.0
        for qid, judgments in results.items():
            for j in judgments:
                print(j.judgment_score)  # 1.0
        ```
    """

    def judge(
        self,
        *,
        query: Document,
        candidates: CandidateGeneratorOutput,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Convert every ``Candidate`` to ``CandidateJudge`` with *judgment_score = 1.0*."""
        result: CandidateJudgeOutput = {}
        for qid, cands in candidates.items():
            judgments = []
            for c in cands:
                judgments.append(CandidateJudge(
                    segment=c.segment,
                    candidate_score=c.score,
                    judgment_score=1.0,
                ))
            result[qid] = judgments
        return result
