# pipeline/generator/_base.py
"""Abstract base class for candidate generators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from locisimiles.document import Document
from locisimiles.pipeline._types import CandidateGeneratorOutput


class CandidateGeneratorBase(ABC):
    """Abstract base class for candidate generators.

    A candidate generator narrows the search space by producing a ranked
    list of source segments for each query segment.  The output is a
    ``CandidateGeneratorOutput`` — a dictionary mapping query-segment IDs
    to lists of ``Candidate`` objects, each containing a source segment
    and a relevance score.

    Subclasses must implement ``generate()``.

    Available implementations:

    - ``EmbeddingCandidateGenerator`` — semantic similarity via sentence
      transformers + ChromaDB.
    - ``ExhaustiveCandidateGenerator`` — returns all query–source pairs
      without filtering.
    - ``RuleBasedCandidateGenerator`` — lexical matching with linguistic
      filters for Latin texts.
    """

    @abstractmethod
    def generate(
        self,
        *,
        query: Document,
        source: Document,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Generate candidate segments from *source* for each query segment.

        Args:
            query: Query document.
            source: Source document.
            **kwargs: Generator-specific parameters.

        Returns:
            Mapping of query segment IDs → lists of ``Candidate`` objects.
        """
        ...
