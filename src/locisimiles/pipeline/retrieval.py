# pipeline/retrieval.py
"""
Retrieval-only pipeline based on embedding similarity.

Provides :class:`RetrievalPipeline` which ranks source segments by
embedding similarity and applies a top-k or threshold criterion to
determine positive matches.
"""
from __future__ import annotations

from typing import Optional

from locisimiles.pipeline.pipeline import Pipeline
from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge.threshold import ThresholdJudge


class RetrievalPipeline(Pipeline):
    """Retrieval pipeline based on embedding similarity.

    Uses a sentence-transformer model to encode query and source segments
    into dense vectors and ranks candidates by cosine similarity.

    Pipeline steps:

    1. **Encoding** - Encode all query and source segments with a
       sentence-transformer model.
    2. **Retrieval** - For each query, retrieve all source segments
       ranked by cosine similarity.
    3. **Thresholding** - Mark the *top_k* most similar candidates as
       positive, or all candidates above *similarity_threshold* if
       provided.

    Args:
        embedding_model_name: HuggingFace model identifier for the
            sentence-transformer.
        device: Torch device string (``"cpu"``, ``"cuda"``, …).
        top_k: Number of top candidates to mark as positive.
        similarity_threshold: If provided, uses threshold instead of top-k.

    Example:
        ```python
        from locisimiles.pipeline import RetrievalPipeline
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Define pipeline
        pipeline = RetrievalPipeline(device="cpu")

        # Run pipeline
        results = pipeline.run(query=query, source=source, top_k=10)
        ```
    """

    def __init__(
        self,
        *,
        embedding_model_name: str = "julian-schelb/SPhilBerta-emb-lat-intertext-v1",
        device: str | int | None = None,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
    ):
        self._device = device if device is not None else "cpu"
        super().__init__(
            generator=EmbeddingCandidateGenerator(
                embedding_model_name=embedding_model_name,
                device=device,
            ),
            judge=ThresholdJudge(
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            ),
        )

    @property
    def device(self) -> str:
        """Device used by the embedding generator."""
        return self.generator.device
