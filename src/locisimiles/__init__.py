"""LociSimiles - Intertextuality detection in Latin literature."""

__version__ = "1.4.0"

from locisimiles.datasets import (
    load_example_ground_truth,
    load_example_query,
    load_example_source,
)
from locisimiles.document import Document, TextSegment
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.pipeline import (
    CandidateJudgeBase,
    ClassificationJudge,
    # Backward-compatible aliases
    ClassificationPipeline,
    ClassificationPipelineWithCandidateGeneration,
    ClassificationPipelineWithCandidategeneration,
    EmbeddingCandidateGenerator,
    ExhaustiveCandidateGenerator,
    ExhaustiveClassificationPipeline,
    IdentityJudge,
    JudgeBase,  # backward-compat alias
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
    "ClassificationPipelineWithCandidateGeneration",
    "ClassificationPipelineWithCandidategeneration",  # old typo kept for compat
    # Modular components
    "Pipeline",
    "CandidateJudgeBase",
    "JudgeBase",  # backward-compat alias
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
    # Example datasets
    "load_example_query",
    "load_example_source",
    "load_example_ground_truth",
]
