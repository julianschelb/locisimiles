"""LociSimiles - Intertextuality detection in Latin literature."""

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline import (
    ClassificationPipeline,
    ClassificationPipelineWithCandidategeneration,
    RetrievalPipeline,
    pretty_print,
)
from locisimiles.evaluator import IntertextEvaluator

__all__ = [
    "Document",
    "TextSegment",
    "ClassificationPipeline",
    "ClassificationPipelineWithCandidategeneration",
    "RetrievalPipeline",
    "IntertextEvaluator",
    "pretty_print",
]
