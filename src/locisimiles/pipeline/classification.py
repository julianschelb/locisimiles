# pipeline/classification.py
"""
Classification-only pipeline for exhaustive pairwise comparison.

Provides ``ClassificationPipeline`` which classifies every possible
query-source pair using a fine-tuned sequence-classification model.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from locisimiles.pipeline.generator.exhaustive import ExhaustiveCandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.pipeline import Pipeline


class ExhaustiveClassificationPipeline(Pipeline):
    """Classification pipeline for exhaustive pairwise comparison.

    For each query segment every source segment is considered as a
    candidate.  Each query-source pair is then fed to a fine-tuned
    sequence-classification model that outputs the probability of the
    pair being an intertextual match.

    Pipeline steps:

    1. **Candidate generation** - Create all possible query-source pairs.
    2. **Classification** - Score each pair with a HuggingFace
       sequence-classification model.  The positive-class probability
       is used as the judgment score.

    Args:
        classification_name: HuggingFace model identifier for the
            sequence-classification model.
        device: Torch device string (``"cpu"``, ``"cuda"``, ...).
        pos_class_idx: Index of the positive class in the classifier output.
        label_names: Optional class labels for binary or multiclass models.
        positive_class_ids: Optional class ids that count as intertextual links.
        positive_labels: Optional labels that count as intertextual links.
        negative_labels: Optional labels that count as non-links.
        emit_class_metadata: Whether to attach predicted labels and class
            probabilities to results.

    Example:
        ```python
        from locisimiles.pipeline import ClassificationPipeline
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Define pipeline
        pipeline = ClassificationPipeline(device="cpu")

        # Run pipeline
        results = pipeline.run(query=query, source=source)
        ```
    """

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,
        label_names: Sequence[str] | Mapping[int | str, str] | None = None,
        positive_class_ids: Sequence[int] | None = None,
        positive_labels: Sequence[str] | None = None,
        negative_labels: Sequence[str] | None = None,
        emit_class_metadata: bool | None = None,
    ):
        super().__init__(
            generator=ExhaustiveCandidateGenerator(),
            judge=ClassificationJudge(
                classification_name=classification_name,
                device=device,
                pos_class_idx=pos_class_idx,
                label_names=label_names,
                positive_class_ids=positive_class_ids,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                emit_class_metadata=emit_class_metadata,
            ),
        )

    @property
    def device(self) -> str:
        """Device used by the classification judge."""
        return self.judge.device


# Backward-compatible alias
ClassificationPipeline = ExhaustiveClassificationPipeline
