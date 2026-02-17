"""LociSimiles - Intertextuality detection in Latin literature."""

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline import (
    # Preconfigured pipelines
    TwoStagePipeline,
    ExhaustiveClassificationPipeline,
    RetrievalPipeline,
    RuleBasedPipeline,
    # Backward-compatible aliases
    ClassificationPipeline,
    ClassificationPipelineWithCandidategeneration,
    # Modular components
    Pipeline,
    EmbeddingCandidateGenerator,
    ExhaustiveCandidateGenerator,
    RuleBasedCandidateGenerator,
    ClassificationJudge,
    ThresholdJudge,
    IdentityJudge,
    # Utilities
    pretty_print,
)
from locisimiles.evaluator import IntertextEvaluator

__all__ = [
    "Document",
    "TextSegment",
    # Preconfigured pipelines
    "TwoStagePipeline",
    "ExhaustiveClassificationPipeline",
    "RetrievalPipeline",
    "RuleBasedPipeline",
    # Backward-compatible aliases
    "ClassificationPipeline",
    "ClassificationPipelineWithCandidategeneration",
    # Modular components
    "Pipeline",
    "EmbeddingCandidateGenerator",
    "ExhaustiveCandidateGenerator",
    "RuleBasedCandidateGenerator",
    "ClassificationJudge",
    "ThresholdJudge",
    "IdentityJudge",
    # Utilities
    "IntertextEvaluator",
    "pretty_print",
]
