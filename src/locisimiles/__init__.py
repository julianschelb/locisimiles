"""LociSimiles - Intertextuality detection in Latin literature."""

__version__ = "1.0.1"

from locisimiles.document import Document, TextSegment
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.pipeline import (
    ClassificationJudge,
    # Backward-compatible aliases
    ClassificationPipeline,
    ClassificationPipelineWithCandidategeneration,
    EmbeddingCandidateGenerator,
    ExhaustiveCandidateGenerator,
    ExhaustiveClassificationPipeline,
    IdentityJudge,
    # Modular components
    Pipeline,
    RetrievalPipeline,
    RuleBasedCandidateGenerator,
    RuleBasedPipeline,
    ThresholdJudge,
    # Preconfigured pipelines
    TwoStagePipeline,
    # Utilities
    pretty_print,
    results_to_csv,
    results_to_json,
)

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
    "results_to_csv",
    "results_to_json",
]
