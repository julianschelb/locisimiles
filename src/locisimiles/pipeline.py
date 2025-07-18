# pipeline.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Sequence
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from locisimiles.document import Document, TextSegment

# ============== UTILITY VARIABLES ==============

ScoreT = float
SimPair = Tuple[TextSegment, ScoreT]              # (segment, cosine-sim)
FullPair = Tuple[TextSegment, ScoreT, ScoreT]      # (+ prob-positive)
SimDict = Dict[str, List[SimPair]]
FullDict = Dict[str, List[FullPair]]

# ============== UTILITY HELPERS ==============


def pretty_print(results: FullDict) -> None:
    """Human-friendly dump of *run()* output."""
    for qid, lst in results.items():
        print(f"\n▶ Query segment {qid!r}:")
        for src_seg, sim, ppos in lst:
            print(f"  {src_seg.id:<25}  sim={sim:+.3f}  P(pos)={ppos:.3f}")

# ============== PIPELINE ==============


class ClassificationPipelineWithCandidategeneration:
    """
    A pipeline for intertextuality classification with candidate generation.
    It first generates candidate segments from a source document based on
    similarity to a query segment, and then classifies these candidates
    as intertextual or not using a pre-trained model.
    """
    
    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-base-latin-intertextuality",
        embedding_model_name: str = "bowphs/SPhilBerta",
        device: str | int | None = None,
        pos_class_idx: int = 1, # Index of the positive class in the classifier
    ):
        self.device = device if device is not None else "cpu"
        self.pos_class_idx = pos_class_idx  

        # -------- Load Models ----------
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classification_name)
        self.clf_model.to(self.device).eval()

        # Keep results in memory for later access
        self._last_sim:  SimDict | None = None
        self._last_full: FullDict | None = None

    # ---------- Generate Embedding ----------

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorise *texts* → normalised float32 numpy array."""
        return self.embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ).astype("float32")

    # ---------- Predict Positive Probability ----------

    def _predict_batch(
        self,
        query_text: str,
        cand_texts: Sequence[str],
    ) -> List[ScoreT]:
        """Predict probabilities for a batch of (query, cand) pairs."""
        encoding = self.clf_tokenizer(
            [query_text] * len(cand_texts),  # Repeat query for each candidate
            cand_texts,  # Candidate texts
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.clf_model(**encoding).logits
            return F.softmax(logits, dim=1)[:, self.pos_class_idx].cpu().tolist()
        
    def _predict(
        self,
        query_text: str,
        cand_texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> List[ScoreT]:
        """Return P(positive) for each (query, cand) pair in *cand_texts*."""
        probs: List[ScoreT] = []
        
        # Predict in batches between a query and multiple candidates
        for i in range(0, len(cand_texts), batch_size):
            chunk = cand_texts[i: i + batch_size]
            chunk_probs = self._predict_batch(query_text, chunk)
            probs.extend(chunk_probs)
        return probs



    # ---------- Stage 1: Retrieval ----------
    
    def _compute_similarity(
        self,
        query_segments: List[TextSegment],
        source_segments: List[TextSegment],
        query_embeddings: np.ndarray,
        source_embeddings: np.ndarray,
        top_k: int,
    ) -> SimDict:
        """
        Compute cosine similarity between query and source embeddings
        and return the top-k similar segments for each query segment.
        """
        # Compute cosine similarity matrix (normalised vectors)
        similarity_matrix = np.matmul(query_embeddings, source_embeddings.T)
        # Initialize a dictionary to store the top-k similar segments for each query segment
        similarity_results: SimDict = {}

        # Iterate over each query segment
        for query_index, query_segment in enumerate(query_segments):
            # Get indices of top-k source segments sorted by similarity (descending order) 
            sorted_indices = np.argsort(similarity_matrix[query_index])[::-1]
            top_indices = sorted_indices[:top_k] if top_k < similarity_matrix.shape[1] else sorted_indices
            # Store the top-k similar segments and their similarity scores
            similarity_results[query_segment.id] = [
                (source_segments[source_index], float(similarity_matrix[query_index, source_index]))
                for source_index in top_indices
            ]

        return similarity_results

    def generate_candidates(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 5,
        **kwargs: Any,
    ) -> SimDict:
        """
        Generate candidate segments from *source* based on similarity to *query*.
        Returns a dictionary mapping query segment IDs to lists of
        (source segment, similarity score) pairs.
        """
        # Extract segments from query and source documents
        query_segments = list(query.segments.values())
        source_segments = list(source.segments.values())

        # Embed query and source segments
        query_embeddings = self._embed([s.text for s in query_segments])
        source_embeddings = self._embed([s.text for s in source_segments])
        
        # Compute similarity between query and source segments
        similarity_results = self._compute_similarity(
            query_segments, source_segments, query_embeddings, source_embeddings, top_k
        )
        
        # Cache the results and return
        self._last_sim = similarity_results
        return similarity_results

    # ---------- Stage 2: Classification ----------

    def check_candidates(
        self,
        *,
        query: Document,
        source: Document,
        candidates: SimDict | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> FullDict:
        """
        Classify candidates generated from *source*.
        If *candidates* is not provided, it will be generated using
        *generate_candidates*.
        Returns a dictionary mapping query segment IDs to lists of
        (source segment, similarity score, P(positive)) tuples.
        """

        full_results: FullDict = {}
        
        for query_id, similarity_pairs in candidates.items():
            
            # Predict probabilities for the current query and its candidates
            candidate_texts = [segment.text for segment, _ in similarity_pairs]
            predicted_probabilities = self._predict(
                query[query_id].text, candidate_texts, batch_size=batch_size
            )
            
            # Combine segments, similarity scores, and probabilities into results
            full_results[query_id] = []
            for (segment, similarity_score), probability in zip(similarity_pairs, predicted_probabilities):
                full_results[query_id].append((segment, similarity_score, probability))

        self._last_full = full_results
        return full_results

    # ---------- Stage 3: Pipeline ----------

    def run(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 5,
        **kwargs: Any,
    ) -> FullDict:
        """
        Run the full pipeline: generate candidates and classify them.
        Returns a dictionary mapping query segment IDs to lists of
        (source segment, similarity score, P(positive)) tuples.
        """
        similarity_dict = self.generate_candidates(
            query=query,
            source=source,
            top_k=top_k,
        )
        return self.check_candidates(
            query=query,
            source=source,
            candidates=similarity_dict,
            **kwargs,
        )

# ================== MAIN ==================


if __name__ == "__main__":

    # Load example query and source documents
    query_doc = Document("../data/hieronymus_samples.csv")
    source_doc = Document("../data/vergil_samples.csv")

    # Load the pipeline with pre-trained models
    pipeline = ClassificationPipelineWithCandidategeneration(
        classification_name="julian-schelb/xlm-roberta-base-latin-intertextuality",
        embedding_model_name="bowphs/SPhilBerta",
        device="cpu",
    )
    
    # Run the pipeline with the query and source documents
    results = pipeline.run(
        query=query_doc, # Query document
        source=source_doc, # Source document
        top_k=3, # Number of top similar candidates to classify
    )
    pretty_print(results)
