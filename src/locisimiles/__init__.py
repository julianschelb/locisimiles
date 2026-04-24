"""LociSimiles - Intertextuality detection in Latin literature."""

__version__ = "1.6.0"

from locisimiles.datasets import (
    load_example_ground_truth,
    load_example_query,
    load_example_source,
)
from locisimiles.document import Document, TextSegment
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.pipeline import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    DEFAULT_WORD2VEC_MODEL_PATH,
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
    LatinBertContextualCandidateGenerator,
    LatinBertRetrievalPipeline,
    LatinBertTwoStagePipeline,
    # Modular components
    Pipeline,
    RetrievalPipeline,
    RuleBasedCandidateGenerator,
    RuleBasedPipeline,
    ThresholdJudge,
    # Preconfigured pipelines
    TwoStagePipeline,
    Word2VecCandidateGenerator,
    Word2VecRetrievalPipeline,
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
    "Word2VecRetrievalPipeline",
    "LatinBertRetrievalPipeline",
    "LatinBertTwoStagePipeline",
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
    "LatinBertContextualCandidateGenerator",
    "Word2VecCandidateGenerator",
    "DEFAULT_CONTEXTUAL_BERT_MODEL_NAME",
    "DEFAULT_WORD2VEC_MODEL_PATH",
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
