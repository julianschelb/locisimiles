# pipeline/__init__.py
"""
Pipeline submodule for intertextuality detection.

This module provides four pipeline implementations:
- ClassificationPipelineWithCandidategeneration: Two-stage (retrieval + classification)
- ClassificationPipeline: Classification-only (exhaustive comparison)
- RetrievalPipeline: Retrieval-only (embedding similarity, no classification)
- RuleBasedPipeline: Rule-based (lexical matching + linguistic filters)

All exports are available at the package level for backward compatibility:
    from locisimiles.pipeline import ClassificationPipeline, pretty_print
"""
from __future__ import annotations

# Import new types for re-export
from locisimiles.pipeline._types import (
    # New dataclasses & type aliases
    Candidate,
    Judgment,
    CandidateGeneratorOutput,
    JudgeInput,
    JudgeOutput,
    # Deprecated backward-compatible aliases
    ScoreT,
    SimPair,
    FullPair,
    SimDict,
    FullDict,
    # Utilities
    pretty_print,
)

# Import pipeline classes
from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
from locisimiles.pipeline.classification import ClassificationPipeline
from locisimiles.pipeline.retrieval import RetrievalPipeline
from locisimiles.pipeline.rule_based import RuleBasedPipeline

# Define public API
__all__ = [
    # New types
    "Candidate",
    "Judgment",
    "CandidateGeneratorOutput",
    "JudgeInput",
    "JudgeOutput",
    # Deprecated aliases (kept for backward compatibility)
    "ScoreT",
    "SimPair",
    "FullPair",
    "SimDict",
    "FullDict",
    # Utilities
    "pretty_print",
    # Pipeline classes
    "ClassificationPipelineWithCandidategeneration",
    "ClassificationPipeline",
    "RetrievalPipeline",
    "RuleBasedPipeline",
]
