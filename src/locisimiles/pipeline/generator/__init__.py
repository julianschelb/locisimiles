# pipeline/generator/__init__.py
"""
Candidate-generator components.

Generators narrow the search space by selecting source segments that
are most likely to be relevant for each query segment.

Available generators:

- ``EmbeddingCandidateGenerator`` — semantic embedding similarity
- ``ExhaustiveCandidateGenerator`` — all pairs (no filtering)
- ``RuleBasedCandidateGenerator`` — lexical matching + linguistic filters
- ``Word2VecCandidateGenerator`` — Burns-style bigram similarity retrieval
"""

from locisimiles.pipeline.generator._base import CandidateGeneratorBase
from locisimiles.pipeline.generator.contextual_bert import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    LatinBertContextualCandidateGenerator,
)
from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
from locisimiles.pipeline.generator.rule_based import RuleBasedCandidateGenerator
from locisimiles.pipeline.generator.word2vec import (
    DEFAULT_WORD2VEC_MODEL_PATH,
    Word2VecCandidateGenerator,
)

__all__ = [
    "CandidateGeneratorBase",
    "LatinBertContextualCandidateGenerator",
    "DEFAULT_CONTEXTUAL_BERT_MODEL_NAME",
    "EmbeddingCandidateGenerator",
    "ExhaustiveCandidateGenerator",
    "RuleBasedCandidateGenerator",
    "Word2VecCandidateGenerator",
    "DEFAULT_WORD2VEC_MODEL_PATH",
]
