# pipeline/retrieval.py
"""
Retrieval-only pipeline: Uses embedding similarity without classification.
"""
from __future__ import annotations

import time
import chromadb
import numpy as np
from typing import Dict, List, Any, Sequence
from sentence_transformers import SentenceTransformer
from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import (
    Candidate,
    CandidateJudge,
    CandidateGeneratorOutput,
    CandidateJudgeOutput,
)
from tqdm import tqdm


class RetrievalPipeline:
    """
    A retrieval-only pipeline for intertextuality detection.
    
    This pipeline uses semantic embeddings to find similar text passages without
    a classification stage. It's useful for fast candidate generation or when
    you want a simpler approach based purely on semantic similarity.
    
    Binary decisions are made using one of two strategies:
        - **Top-k**: The k most similar candidates are marked as positive
        - **Similarity threshold**: Candidates above a threshold are positive
    
    The results are returned in ``CandidateJudgeOutput`` format for compatibility with
    the evaluator, where ``judgment_score`` is 1.0 for positive and 0.0 for
    negative.
    
    Attributes:
        embedder: The sentence transformer model for computing embeddings.
        device: The device used for computation ('cpu' or 'cuda').
    
    Example:
        ```python
        from locisimiles.pipeline import RetrievalPipeline
        from locisimiles.document import Document
        
        # Load documents
        query_doc = Document("hieronymus.csv")
        source_doc = Document("vergil.csv")
        
        # Initialize pipeline
        pipeline = RetrievalPipeline(
            embedding_model_name="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
            device="cpu",
        )
        
        # Find top 5 similar passages for each query
        results = pipeline.run(
            query=query_doc,
            source=source_doc,
            top_k=5,
        )
        ```
    """

    def __init__(
        self,
        *,
        embedding_model_name: str = "julian-schelb/SPhilBerta-emb-lat-intertext-v1",
        device: str | int | None = None,
    ):
        self.device = device if device is not None else "cpu"
        self._source_index: chromadb.Collection | None = None

        # -------- Load Embedding Model ----------
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)

        # Keep results in memory for later access
        self._last_candidates: CandidateGeneratorOutput | None = None
        self._last_judgments: CandidateJudgeOutput | None = None

    # ---------- Generate Embedding ----------

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
    ):
        """Create a Chroma collection from *source_segments* and their embeddings."""
        
        client = chromadb.EphemeralClient()
        unique_name = f"{collection_name}_{int(time.time() * 1000000)}"
        
        try:
            client.delete_collection(name=unique_name)
        except Exception:
            pass
        
        col = client.create_collection(
            name=unique_name,
            metadata={"hnsw:space": "cosine"}
        )

        ids = [s.id for s in source_segments]
        embeddings = source_embeddings.tolist()

        for i in range(0, len(ids), batch_size):
            col.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
            )

        return col
    
    def _compute_similarity(
        self,
        query_segments: List[TextSegment],
        query_embeddings: np.ndarray,
        source_document: Document,
        top_k: int,
    ) -> CandidateGeneratorOutput:
        """
        Compute cosine similarity between query embeddings and source embeddings
        using the Chroma index, and return the top-k similar segments for each query segment.
        """
        similarity_results: CandidateGeneratorOutput = {}

        for query_segment, query_embedding in zip(query_segments, query_embeddings):
            results = self._source_index.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
            
            # Convert cosine distance to cosine similarity: similarity = 1 - distance
            similarity_results[query_segment.id] = [
                Candidate(segment=source_document[idx], score=1.0 - float(distance))
                for idx, distance in zip(results["ids"][0], results["distances"][0])
            ]

        return similarity_results

    # ---------- Retrieval ----------

    def retrieve(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 100,
        query_prompt_name: str = "query",
        source_prompt_name: str = "match",
        **kwargs: Any,
    ) -> CandidateGeneratorOutput:
        """
        Retrieve candidate segments from *source* based on similarity to *query*.

        Returns:
            CandidateGeneratorOutput mapping query segment IDs to lists of
            ``Candidate`` objects, sorted by score (descending).
        """
        query_segments = list(query.segments.values())
        source_segments = list(source.segments.values())

        query_embeddings = self._embed(
            [s.text for s in tqdm(query_segments, desc="Embedding query segments")],
            prompt_name=query_prompt_name
        )
        
        source_embeddings = self._embed(
            [s.text for s in tqdm(source_segments, desc="Embedding source segments")],
            prompt_name=source_prompt_name
        )
        
        self._source_index = self.build_source_index(
            source_segments=source_segments,
            source_embeddings=source_embeddings,
            collection_name="source_segments",
        )
        
        similarity_results = self._compute_similarity(
            query_segments=query_segments,
            query_embeddings=query_embeddings,
            source_document=source,
            top_k=top_k,
        )
        
        self._last_candidates = similarity_results
        return similarity_results

    # ---------- Main Pipeline ----------

    def run(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 10,
        similarity_threshold: float | None = None,
        query_prompt_name: str = "query",
        source_prompt_name: str = "match",
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """
        Run the retrieval pipeline and return results compatible with the evaluator.

        Binary decisions are made using one of two criteria:
        - **top_k** (default): The top-k ranked candidates per query are predicted
          as positive (``judgment_score=1.0``), all others as negative (``0.0``).
        - **similarity_threshold**: If provided, candidates with similarity >=
          threshold are predicted as positive, regardless of rank.

        Returns:
            CandidateJudgeOutput mapping query IDs to lists of ``CandidateJudge``
            objects.
        """
        # Retrieve more candidates than top_k to ensure we have enough for evaluation
        # When using similarity_threshold, we need all candidates
        retrieve_k = len(source) if similarity_threshold is not None else top_k
        
        candidate_dict = self.retrieve(
            query=query,
            source=source,
            top_k=retrieve_k,
            query_prompt_name=query_prompt_name,
            source_prompt_name=source_prompt_name,
        )
        
        # Convert to CandidateJudgeOutput format with binary judgment scores
        judge_results: CandidateJudgeOutput = {}
        
        for query_id, candidate_list in candidate_dict.items():
            judge_results[query_id] = []
            
            for rank, candidate in enumerate(candidate_list):
                # Determine if this candidate should be predicted as positive
                if similarity_threshold is not None:
                    is_positive = candidate.score >= similarity_threshold
                else:
                    is_positive = rank < top_k
                
                judge_results[query_id].append(CandidateJudge(
                    segment=candidate.segment,
                    candidate_score=candidate.score,
                    judgment_score=1.0 if is_positive else 0.0,
                ))
        
        self._last_judgments = judge_results
        return judge_results
