# pipeline/__init__.py
"""
Pipeline submodule for intertextuality detection.

This module provides three pipeline implementations:
- ClassificationPipelineWithCandidategeneration: Two-stage (retrieval + classification)
- ClassificationPipeline: Classification-only (exhaustive comparison)
- RetrievalPipeline: Retrieval-only (embedding similarity, no classification)

All exports are available at the package level for backward compatibility:
    from locisimiles.pipeline import ClassificationPipeline, pretty_print
"""
from __future__ import annotations

# Import types for re-export
from locisimiles.pipeline._types import (
    ScoreT,
    SimPair,
    FullPair,
    SimDict,
    FullDict,
    pretty_print,
)

# Import pipeline classes
from locisimiles.pipeline.two_stage import ClassificationPipelineWithCandidategeneration
from locisimiles.pipeline.classification import ClassificationPipeline
from locisimiles.pipeline.retrieval import RetrievalPipeline

# Define public API
__all__ = [
    # Types
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
]
