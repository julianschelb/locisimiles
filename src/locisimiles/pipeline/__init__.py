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
- ``RuleBasedPipeline``: Rule-based lexical matching + linguistic filters

All exports are available at the package level::

    from locisimiles.pipeline import TwoStagePipeline, pretty_print
"""
from __future__ import annotations

# --- Types ---
from locisimiles.pipeline._types import (
    # New dataclasses & type aliases
    Candidate,
    CandidateJudge,
    CandidateGeneratorOutput,
    CandidateJudgeInput,
    CandidateJudgeOutput,
    # Deprecated backward-compatible aliases
    Judgment,
    JudgeInput,
    JudgeOutput,
    ScoreT,
    SimPair,
    FullPair,
    SimDict,
    FullDict,
    # Utilities
    pretty_print,
)

# --- Modular components: generators ---
from locisimiles.pipeline.generator import (
    CandidateGeneratorBase,
    EmbeddingCandidateGenerator,
    ExhaustiveCandidateGenerator,
    RuleBasedCandidateGenerator,
)

# --- Modular components: judges ---
from locisimiles.pipeline.judge import (
    JudgeBase,
    ClassificationJudge,
    ThresholdJudge,
    IdentityJudge,
)

# --- Pipeline composer ---
from locisimiles.pipeline.pipeline import Pipeline

# --- Preconfigured pipelines ---
from locisimiles.pipeline.two_stage import (
    TwoStagePipeline,
    ClassificationPipelineWithCandidategeneration,  # backward-compat alias
)
from locisimiles.pipeline.classification import (
    ExhaustiveClassificationPipeline,
    ClassificationPipeline,  # backward-compat alias
)
from locisimiles.pipeline.retrieval import RetrievalPipeline

# --- Rule-based pipeline ---
from locisimiles.pipeline.rule_based import RuleBasedPipeline

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
    # Generators
    "CandidateGeneratorBase",
    "EmbeddingCandidateGenerator",
    "ExhaustiveCandidateGenerator",
    "RuleBasedCandidateGenerator",
    # Judges
    "JudgeBase",
    "ClassificationJudge",
    "ThresholdJudge",
    "IdentityJudge",
    # Pipeline composer
    "Pipeline",
    # Preconfigured pipelines
    "TwoStagePipeline",
    "ExhaustiveClassificationPipeline",
    "RetrievalPipeline",
    "RuleBasedPipeline",
    # Backward-compatible aliases
    "ClassificationPipelineWithCandidategeneration",
    "ClassificationPipeline",
]
