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
from locisimiles.pipeline._types import SimDict, FullDict
from tqdm import tqdm


class RetrievalPipeline:
    """
    A retrieval-only pipeline for intertextuality detection.
    
    Uses embedding similarity to find candidate segments without a classification
    stage. Binary decisions are made based on rank (top-k) or similarity threshold.
    
    To maintain compatibility with the evaluator, results are returned in FullDict
    format where the "probability" field is set to 1.0 for positive predictions
    and 0.0 for negative predictions based on the decision criteria.
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
        self._last_sim: SimDict | None = None
        self._last_full: FullDict | None = None

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
    ) -> SimDict:
        """
        Compute cosine similarity between query embeddings and source embeddings
        using the Chroma index, and return the top-k similar segments for each query segment.
        """
        similarity_results: SimDict = {}

        for query_segment, query_embedding in zip(query_segments, query_embeddings):
            results = self._source_index.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
            
            # Convert cosine distance to cosine similarity: similarity = 1 - distance
            similarity_results[query_segment.id] = [
                (source_document[idx], 1.0 - float(distance))
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
    ) -> SimDict:
        """
        Retrieve candidate segments from *source* based on similarity to *query*.
        
        Returns a dictionary mapping query segment IDs to lists of
        (source segment, similarity score) pairs, sorted by similarity (descending).
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
        
        self._last_sim = similarity_results
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
    ) -> FullDict:
        """
        Run the retrieval pipeline and return results compatible with the evaluator.
        
        Binary decisions are made using one of two criteria:
        - **top_k** (default): The top-k ranked candidates per query are predicted 
          as positive (prob=1.0), all others as negative (prob=0.0).
        - **similarity_threshold**: If provided, candidates with similarity >= threshold
          are predicted as positive, regardless of rank.
        
        Args:
            query: Query document
            source: Source document  
            top_k: Number of top candidates to mark as positive (default: 10).
                   Used when similarity_threshold is None.
            similarity_threshold: If provided, use this similarity cutoff instead
                   of top_k. Candidates with similarity >= threshold are positive.
            query_prompt_name: Prompt name for query embeddings
            source_prompt_name: Prompt name for source embeddings
            
        Returns:
            FullDict mapping query IDs to lists of (segment, similarity, probability)
            tuples. Probability is 1.0 for positive predictions, 0.0 for negative.
        """
        # Retrieve more candidates than top_k to ensure we have enough for evaluation
        # When using similarity_threshold, we need all candidates
        retrieve_k = len(source) if similarity_threshold is not None else top_k
        
        similarity_dict = self.retrieve(
            query=query,
            source=source,
            top_k=retrieve_k,
            query_prompt_name=query_prompt_name,
            source_prompt_name=source_prompt_name,
        )
        
        # Convert to FullDict format with binary "probabilities"
        full_results: FullDict = {}
        
        for query_id, similarity_pairs in similarity_dict.items():
            full_results[query_id] = []
            
            for rank, (segment, similarity) in enumerate(similarity_pairs):
                # Determine if this candidate should be predicted as positive
                if similarity_threshold is not None:
                    # Use similarity threshold
                    is_positive = similarity >= similarity_threshold
                else:
                    # Use top-k ranking (0-indexed, so rank < top_k)
                    is_positive = rank < top_k
                
                # Set probability to 1.0 for positive, 0.0 for negative
                probability = 1.0 if is_positive else 0.0
                full_results[query_id].append((segment, similarity, probability))
        
        self._last_full = full_results
        return full_results
