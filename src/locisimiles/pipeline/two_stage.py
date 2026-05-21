# pipeline/two_stage.py
"""
Two-stage pipeline: Embedding retrieval followed by classification.

Provides ``ClassificationPipelineWithCandidategeneration`` which first
narrows down candidates using embedding similarity and then classifies the
remaining pairs with a fine-tuned sequence-classification model.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.pipeline import Pipeline


class TwoStagePipeline(Pipeline):
    """Two-stage pipeline: embedding retrieval + classification.

    Combines a fast embedding-based retrieval step with a more expensive
    classification step to efficiently identify intertextual parallels
    in large corpora.

    Pipeline steps:

    1. **Retrieval** - Encode all segments with a sentence-transformer
       model and retrieve the *top_k* most similar source segments for
       each query segment using cosine similarity.
    2. **Classification** - Feed each query-candidate pair into a
       fine-tuned sequence-classification model.  The positive-class
       probability is used as the judgment score.

    Args:
        classification_name: HuggingFace model identifier for the
            sequence-classification model.
        embedding_model_name: HuggingFace model identifier for the
            sentence-transformer.
        device: Torch device string (``"cpu"``, ``"cuda"``, …).
        pos_class_idx: Index of the positive class in the classifier output.
        label_names: Optional class labels for binary or multiclass models.
        positive_class_ids: Optional class ids that count as intertextual links.
        positive_labels: Optional labels that count as intertextual links.
        negative_labels: Optional labels that count as non-links.
        emit_class_metadata: Whether to attach predicted labels and class
            probabilities to results.

    Example:
        ```python
        from locisimiles.pipeline import ClassificationPipelineWithCandidategeneration
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Define pipeline
        pipeline = ClassificationPipelineWithCandidategeneration(device="cpu")

        # Run pipeline
        results = pipeline.run(query=query, source=source, top_k=10)
        ```
    """

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
        embedding_model_name: str = "julian-schelb/multilingual-e5-large-emb-lat-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,
        label_names: Sequence[str] | Mapping[int | str, str] | None = None,
        positive_class_ids: Sequence[int] | None = None,
        positive_labels: Sequence[str] | None = None,
        negative_labels: Sequence[str] | None = None,
        emit_class_metadata: bool | None = None,
    ):
        super().__init__(
            generator=EmbeddingCandidateGenerator(
                embedding_model_name=embedding_model_name,
                device=device,
            ),
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
ClassificationPipelineWithCandidateGeneration = TwoStagePipeline
"""Correctly-cased alias for ``TwoStagePipeline``."""

# Backward-compatible alias (old lowercase typo kept for compatibility)
ClassificationPipelineWithCandidategeneration = TwoStagePipeline
