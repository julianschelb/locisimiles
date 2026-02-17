# pipeline/two_stage.py
"""
Two-stage pipeline: Embedding retrieval followed by classification.

Provides :class:`ClassificationPipelineWithCandidategeneration` which first
narrows down candidates using embedding similarity and then classifies the
remaining pairs with a fine-tuned sequence-classification model.
"""
from __future__ import annotations

from locisimiles.pipeline.pipeline import Pipeline
from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge.classification import ClassificationJudge


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
        classification_name: str = "julian-schelb/PhilBerta-class-latin-intertext-v1",
        embedding_model_name: str = "julian-schelb/SPhilBerta-emb-lat-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,
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
            ),
        )

    @property
    def device(self) -> str:
        """Device used by the classification judge."""
        return self.judge.device


# Backward-compatible alias
ClassificationPipelineWithCandidategeneration = TwoStagePipeline
