# pipeline/__init__.py
"""
Pipeline submodule for intertextuality detection.

This module provides:

**Modular components** (recommended for new code):

- Generators: ``EmbeddingCandidateGenerator``, ``ExhaustiveCandidateGenerator``,
  ``RuleBasedCandidateGenerator``
- Judges: ``ClassificationJudge``, ``ThresholdJudge``, ``IdentityJudge``
- Pipeline: Generic ``Pipeline(generator, judge)`` composer

**Preconfigured pipelines** (convenience wrappers):

- ``TwoStagePipeline``: Embedding retrieval + classification
- ``ExhaustiveClassificationPipeline``: Exhaustive pairs + classification
- ``RetrievalPipeline``: Embedding retrieval + threshold judge
- ``Word2VecRetrievalPipeline``: Burns-style Word2Vec retrieval + threshold judge
- ``TfidfRetrievalPipeline``: TF-IDF retrieval + threshold judge
- ``BM25RetrievalPipeline``: BM25 retrieval + threshold judge
- ``BM25TwoStagePipeline``: BM25 retrieval + classification ("best combined")
- ``BM25LexicalTwoStagePipeline``: BM25 retrieval + lexical classification ("best non-neural")
- ``RuleBasedPipeline``: Rule-based lexical matching + linguistic filters

All exports are available at the package level::

    from locisimiles.pipeline import TwoStagePipeline, pretty_print
"""

from __future__ import annotations

from locisimiles.pipeline._types import (
    # New dataclasses & type aliases
    Candidate,
    CandidateGeneratorOutput,
    CandidateJudge,
    CandidateJudgeInput,
    CandidateJudgeOutput,
    FullDict,
    FullPair,
    JudgeInput,
    JudgeOutput,
    # Deprecated backward-compatible aliases
    Judgment,
    ScoreT,
    SimDict,
    SimPair,
    # Utilities
    pretty_print,
    results_to_csv,
    results_to_json,
)
from locisimiles.pipeline.bm25 import BM25RetrievalPipeline
from locisimiles.pipeline.bm25_lexical_two_stage import BM25LexicalTwoStagePipeline
from locisimiles.pipeline.bm25_two_stage import BM25TwoStagePipeline
from locisimiles.pipeline.classification import (
    ClassificationPipeline,  # backward-compat alias
    ExhaustiveClassificationPipeline,
)
from locisimiles.pipeline.contextual_retrieval import LatinBertRetrievalPipeline
from locisimiles.pipeline.contextual_two_stage import LatinBertTwoStagePipeline
from locisimiles.pipeline.generator import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    DEFAULT_WORD2VEC_MODEL_PATH,
    BM25CandidateGenerator,
    CandidateGeneratorBase,
    EmbeddingCandidateGenerator,
    ExhaustiveCandidateGenerator,
    LatinBertContextualCandidateGenerator,
    RuleBasedCandidateGenerator,
    TfidfCandidateGenerator,
    Word2VecCandidateGenerator,
)
from locisimiles.pipeline.judge import (
    CandidateJudgeBase,
    ClassificationJudge,
    IdentityJudge,
    JudgeBase,  # backward-compat alias
    LexicalClassifierJudge,
    ThresholdJudge,
)
from locisimiles.pipeline.pipeline import Pipeline
from locisimiles.pipeline.retrieval import RetrievalPipeline
from locisimiles.pipeline.rule_based import RuleBasedPipeline
from locisimiles.pipeline.tfidf import TfidfRetrievalPipeline
from locisimiles.pipeline.two_stage import (
    ClassificationPipelineWithCandidateGeneration,  # correctly-cased alias
    ClassificationPipelineWithCandidategeneration,  # backward-compat alias (old typo)
    TwoStagePipeline,
)
from locisimiles.pipeline.word2vec import Word2VecRetrievalPipeline

# Define public API
__all__ = [
    # Types
    "Candidate",
    "CandidateJudge",
    "CandidateGeneratorOutput",
    "CandidateJudgeInput",
    "CandidateJudgeOutput",
    # Deprecated aliases (kept for backward compatibility)
    "Judgment",
    "JudgeInput",
    "JudgeOutput",
    "ScoreT",
    "SimPair",
    "FullPair",
    "SimDict",
    "FullDict",
    # Utilities
    "pretty_print",
    "results_to_csv",
    "results_to_json",
    # Generators
    "CandidateGeneratorBase",
    "EmbeddingCandidateGenerator",
    "ExhaustiveCandidateGenerator",
    "RuleBasedCandidateGenerator",
    "Word2VecCandidateGenerator",
    "DEFAULT_WORD2VEC_MODEL_PATH",
    "LatinBertContextualCandidateGenerator",
    "DEFAULT_CONTEXTUAL_BERT_MODEL_NAME",
    "TfidfCandidateGenerator",
    "BM25CandidateGenerator",
    # Judges
    "CandidateJudgeBase",
    "JudgeBase",  # backward-compat alias
    "ClassificationJudge",
    "LexicalClassifierJudge",
    "ThresholdJudge",
    "IdentityJudge",
    # Pipeline composer
    "Pipeline",
    # Preconfigured pipelines
    "TwoStagePipeline",
    "ExhaustiveClassificationPipeline",
    "RetrievalPipeline",
    "Word2VecRetrievalPipeline",
    "TfidfRetrievalPipeline",
    "BM25RetrievalPipeline",
    "BM25TwoStagePipeline",
    "BM25LexicalTwoStagePipeline",
    "LatinBertRetrievalPipeline",
    "LatinBertTwoStagePipeline",
    "RuleBasedPipeline",
    # Backward-compatible aliases
    "ClassificationPipelineWithCandidateGeneration",
    "ClassificationPipelineWithCandidategeneration",  # old typo kept for compat
    "ClassificationPipeline",
]
