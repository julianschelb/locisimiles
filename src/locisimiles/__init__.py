"""LociSimiles - Intertextuality detection in Latin literature."""

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline import (
    ClassificationPipeline,
    ClassificationPipelineWithCandidategeneration,
    RetrievalPipeline,
    RuleBasedPipeline,
    pretty_print,
)
from locisimiles.evaluator import IntertextEvaluator

__all__ = [
    "Document",
    "TextSegment",
    "ClassificationPipeline",
    "ClassificationPipelineWithCandidategeneration",
    "RetrievalPipeline",
    "RuleBasedPipeline",
    "IntertextEvaluator",
    "pretty_print",
]
