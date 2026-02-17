# pipeline/generator/rule_based.py
"""Rule-based candidate generator — adapter around RuleBasedPipeline."""
from __future__ import annotations

from typing import Any

from locisimiles.document import Document
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase


class RuleBasedCandidateGenerator(CandidateGeneratorBase):
    """Generate candidates using lexical matching and linguistic filters.

    This is a thin adapter around
    :class:`~locisimiles.pipeline.rule_based.RuleBasedPipeline`.  It
    delegates to the rule-based pipeline's matching logic and converts the
    output to ``CandidateGeneratorOutput``.

    Constructor arguments are forwarded to ``RuleBasedPipeline``.

    Example:
        ```python
        from locisimiles.pipeline.generator import RuleBasedCandidateGenerator
        from locisimiles.document import Document

        generator = RuleBasedCandidateGenerator(min_shared_words=3)
        candidates = generator.generate(
            query=Document("query.csv"),
            source=Document("source.csv"),
            query_genre="prose",
            source_genre="poetry",
        )
        ```
    """

    def __init__(self, **kwargs: Any):
        from locisimiles.pipeline.rule_based import RuleBasedPipeline

        self._pipeline = RuleBasedPipeline(**kwargs)

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Run rule-based matching and return candidates.

        Extra *kwargs* (e.g. ``query_genre``, ``source_genre``) are
        forwarded to :meth:`RuleBasedPipeline.run`.

        Args:
            query: Query document.
            source: Source document.

        Returns:
            Mapping of query segment IDs → lists of ``Candidate``.
        """
        judge_output = self._pipeline.run(query=query, source=source, **kwargs)

        result: CandidateGeneratorOutput = {}
        for qid, judgments in judge_output.items():
            candidates = []
            for j in judgments:
                score = j.candidate_score if j.candidate_score is not None else 0.0
                candidates.append(Candidate(segment=j.segment, score=score))
            result[qid] = candidates
        return result
