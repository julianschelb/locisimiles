# pipeline/generator/exhaustive.py
"""Exhaustive candidate generator — returns all query×source pairs."""

from __future__ import annotations

from typing import Any

from locisimiles.document import Document
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase


class ExhaustiveCandidateGenerator(CandidateGeneratorBase):
    """Treat every source segment as a candidate for every query segment.

    No scoring or ranking is performed.  Each ``Candidate.score`` is set
    to ``1.0`` since all pairs are treated equally.

    This generator is typically paired with a judge
    (e.g. ``ClassificationJudge``) that performs the actual scoring.
    Best suited for smaller datasets where comparing all pairs is
    feasible.

    Example:
        ```python
        from locisimiles.pipeline.generator import ExhaustiveCandidateGenerator
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Generate all possible pairs
        generator = ExhaustiveCandidateGenerator()
        candidates = generator.generate(query=query, source=source)

        # Total pairs = len(query) × len(source)
        total = sum(len(c) for c in candidates.values())
        print(f"{total} candidate pairs")
        ```
    """

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Return all source segments as candidates for each query segment.

        Args:
            query: Query document.
            source: Source document.

        Returns:
            Mapping of query segment IDs → lists of ``Candidate`` with
            ``score=1.0``.
        """
        source_segments = list(source.segments.values())

        result: CandidateGeneratorOutput = {}
        for query_seg in query.segments.values():
            candidates = []
            for src_seg in source_segments:
                candidates.append(Candidate(segment=src_seg, score=1.0))
            result[query_seg.id] = candidates
        return result
