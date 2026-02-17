# pipeline/generator/__init__.py
"""
Candidate-generator components.

Generators narrow the search space by selecting source segments that
are most likely to be relevant for each query segment.

Available generators:

- :class:`EmbeddingCandidateGenerator` — semantic embedding similarity
- :class:`ExhaustiveCandidateGenerator` — all pairs (no filtering)
- :class:`RuleBasedCandidateGenerator` — lexical matching + linguistic filters
"""
from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator

__all__ = [
    "CandidateGeneratorBase",
    "EmbeddingCandidateGenerator",
    "ExhaustiveCandidateGenerator",
    "RuleBasedCandidateGenerator",
]
