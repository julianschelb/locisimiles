# pipeline/generator/embedding.py
"""Embedding-based candidate generator using sentence transformers and ChromaDB."""

from __future__ import annotations

import time
from typing import Any, List, Sequence

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import Candidate, CandidateGeneratorOutput
from locisimiles.pipeline.generator._base import CandidateGeneratorBase


class EmbeddingCandidateGenerator(CandidateGeneratorBase):
    """Generate candidates using semantic embedding similarity.

    Encodes query and source segments with a sentence-transformer model,
    builds an ephemeral ChromaDB index on the source embeddings, and
    retrieves the most similar source segments for each query segment.

    The number of candidates per query is controlled by the ``top_k``
    parameter passed to ``generate()``.

    Args:
        embedding_model_name: HuggingFace model identifier for the
            sentence-transformer.  Defaults to the pre-trained Latin
            intertextuality model.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).

    Example:
        ```python
        from locisimiles.pipeline.generator import EmbeddingCandidateGenerator
        from locisimiles.document import Document

        # Load documents
        query = Document("query.csv")
        source = Document("source.csv")

        # Generate candidates
        generator = EmbeddingCandidateGenerator(device="cpu")
        candidates = generator.generate(query=query, source=source, top_k=10)

        # candidates is a dict: {query_id: [Candidate, ...]}
        for query_id, cands in candidates.items():
            print(f"{query_id}: {len(cands)} candidates")
        ```
    """

    def __init__(
        self,
        *,
        embedding_model_name: str = "julian-schelb/SPhilBerta-emb-lat-intertext-v1",
        device: str | int | None = None,
    ):
        self.device = device if device is not None else "cpu"
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self._source_index: chromadb.Collection | None = None

    # ---------- Embedding ----------

    def _embed(self, texts: Sequence[str], prompt_name: str) -> np.ndarray:
        """Vectorise *texts* → normalised float32 numpy array."""
        return self.embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
            prompt_name=prompt_name if prompt_name else None,
        ).astype("float32")

    # ---------- Index Building ----------

    def build_source_index(
        self,
        source_segments: Sequence[TextSegment],
        source_embeddings: np.ndarray,
        collection_name: str = "source_segments",
        batch_size: int = 5000,
    ) -> chromadb.Collection:
        """Create an ephemeral Chroma collection from segments and embeddings."""
        client = chromadb.EphemeralClient()
        unique_name = f"{collection_name}_{int(time.time() * 1000000)}"

        import contextlib

        with contextlib.suppress(Exception):
            client.delete_collection(name=unique_name)

        col = client.create_collection(
            name=unique_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [s.id for s in source_segments]
        embeddings = source_embeddings.tolist()

        for i in range(0, len(ids), batch_size):
            col.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
            )

        return col

    # ---------- Similarity ----------

    def _compute_similarity(
        self,
        query_segments: List[TextSegment],
        query_embeddings: np.ndarray,
        source_document: Document,
        top_k: int,
    ) -> CandidateGeneratorOutput:
        """Query ChromaDB index and return top-k candidates per query segment."""
        similarity_results: CandidateGeneratorOutput = {}

        for query_segment, query_embedding in zip(query_segments, query_embeddings):
            results = self._source_index.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            candidates = []
            for idx, distance in zip(results["ids"][0], results["distances"][0]):
                candidates.append(
                    Candidate(
                        segment=source_document[idx],
                        score=1.0 - float(distance),
                    )
                )
            similarity_results[query_segment.id] = candidates

        return similarity_results

    # ---------- CandidateGeneratorBase ----------

    def generate(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 100,
        query_prompt_name: str = "query",
        source_prompt_name: str = "match",
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """Generate candidates by embedding similarity.

        Encodes all segments, indexes the source embeddings, and returns
        the ``top_k`` most similar source segments for each query segment.

        Args:
            query: Query document.
            source: Source document.
            top_k: Number of most-similar source segments to return per
                query segment.
            query_prompt_name: Prompt name passed to the sentence-transformer
                for query encoding.
            source_prompt_name: Prompt name passed to the sentence-transformer
                for source encoding.

        Returns:
            Mapping of query segment IDs → ranked lists of ``Candidate``
            sorted by descending cosine similarity.
        """
        query_segments = list(query.segments.values())
        source_segments = list(source.segments.values())

        query_embeddings = self._embed(
            [s.text for s in tqdm(query_segments, desc="Embedding query segments")],
            prompt_name=query_prompt_name,
        )

        source_embeddings = self._embed(
            [s.text for s in tqdm(source_segments, desc="Embedding source segments")],
            prompt_name=source_prompt_name,
        )

        self._source_index = self.build_source_index(
            source_segments=source_segments,
            source_embeddings=source_embeddings,
        )

        return self._compute_similarity(
            query_segments=query_segments,
            query_embeddings=query_embeddings,
            source_document=source,
            top_k=top_k,
        )
