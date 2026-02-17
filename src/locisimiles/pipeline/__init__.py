# pipeline/__init__.py
"""
Pipeline submodule for intertextuality detection.

This module provides:

**Modular components** (recommended for new code):

- Generators: ``EmbeddingCandidateGenerator``, ``ExhaustiveCandidateGenerator``,
  ``RuleBasedCandidateGenerator``
- Judges: ``ClassificationJudge``, ``ThresholdJudge``, ``IdentityJudge``
- Pipeline: Generic ``Pipeline(generator, judge)`` composer

**Legacy pipeline classes** (backward-compatible):

- ``ClassificationPipelineWithCandidategeneration``: Two-stage (retrieval + classification)
- ``ClassificationPipeline``: Classification-only (exhaustive comparison)
- ``RetrievalPipeline``: Retrieval-only (embedding similarity, no classification)
- ``RuleBasedPipeline``: Rule-based (lexical matching + linguistic filters)

All exports are available at the package level for backward compatibility::

    from locisimiles.pipeline import ClassificationPipeline, pretty_print
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

# --- Legacy pipeline classes ---
from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
from locisimiles.pipeline.classification import ClassificationPipeline
from locisimiles.pipeline.retrieval import RetrievalPipeline
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
    # Legacy pipeline classes
    "ClassificationPipelineWithCandidategeneration",
    "ClassificationPipeline",
    "RetrievalPipeline",
    "RuleBasedPipeline",
]
