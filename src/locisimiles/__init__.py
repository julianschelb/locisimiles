"""LociSimiles - Intertextuality detection in Latin literature."""

__version__ = "2.0.0"

from locisimiles.datasets import (
    load_example_ground_truth,
    load_example_query,
    load_example_source,
)
from locisimiles.document import Document, TextSegment
from locisimiles.evaluator import IntertextEvaluator
from locisimiles.ground_truth import GroundTruth, GroundTruthEntry
from locisimiles.pipeline import (
    DEFAULT_CONTEXTUAL_BERT_MODEL_NAME,
    DEFAULT_WORD2VEC_MODEL_PATH,
    BM25CandidateGenerator,
    BM25LexicalTwoStagePipeline,
    BM25RetrievalPipeline,
    BM25TwoStagePipeline,
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
    LexicalClassifierJudge,
    # Modular components
    Pipeline,
    RetrievalPipeline,
    RuleBasedCandidateGenerator,
    RuleBasedPipeline,
    TfidfCandidateGenerator,
    TfidfRetrievalPipeline,
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
    "GroundTruth",
    "GroundTruthEntry",
    # Preconfigured pipelines
    "TwoStagePipeline",
    "ExhaustiveClassificationPipeline",
    "RetrievalPipeline",
    "Word2VecRetrievalPipeline",
    "LatinBertRetrievalPipeline",
    "LatinBertTwoStagePipeline",
    "TfidfRetrievalPipeline",
    "BM25RetrievalPipeline",
    "BM25TwoStagePipeline",
    "BM25LexicalTwoStagePipeline",
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
    "TfidfCandidateGenerator",
    "BM25CandidateGenerator",
    "DEFAULT_CONTEXTUAL_BERT_MODEL_NAME",
    "DEFAULT_WORD2VEC_MODEL_PATH",
    "ClassificationJudge",
    "LexicalClassifierJudge",
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
