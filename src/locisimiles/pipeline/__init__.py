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
from locisimiles.pipeline.classification import (
    ClassificationPipeline,  # backward-compat alias
    ExhaustiveClassificationPipeline,
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
    ClassificationJudge,
    IdentityJudge,
    JudgeBase,
    ThresholdJudge,
)

# --- Pipeline composer ---
from locisimiles.pipeline.pipeline import Pipeline
from locisimiles.pipeline.retrieval import RetrievalPipeline

# --- Rule-based pipeline ---
from locisimiles.pipeline.rule_based import RuleBasedPipeline

# --- Preconfigured pipelines ---
from locisimiles.pipeline.two_stage import (
    ClassificationPipelineWithCandidategeneration,  # backward-compat alias
    TwoStagePipeline,
)

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
