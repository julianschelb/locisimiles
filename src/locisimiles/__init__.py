"""LociSimiles - Intertextuality detection in Latin literature."""

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline import (
    # Legacy pipeline classes
    ClassificationPipeline,
    ClassificationPipelineWithCandidategeneration,
    RetrievalPipeline,
    RuleBasedPipeline,
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
    # Legacy pipeline classes
    "ClassificationPipeline",
    "ClassificationPipelineWithCandidategeneration",
    "RetrievalPipeline",
    "RuleBasedPipeline",
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
