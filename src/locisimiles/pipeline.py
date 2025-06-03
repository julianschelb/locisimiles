# pipeline.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Sequence

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from document import Document, TextSegment

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

    POS_CLASS_IDX = 1  # index of the "positive / intertextual" label

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-base-latin-intertextuality",
        embedding_model_name: str = "bowphs/PhilBerta",
        device: str | int | None = None,
    ):
        self.device = device if device is not None else "cpu"

        # -------- Models ----------
        self.embedder = SentenceTransformer(
            embedding_model_name, device=self.device)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(
            classification_name
        ).to(self.device).eval()

        # Last results for inspection
        self._last_sim:  SimDict | None = None
        self._last_full: FullDict | None = None

    # ---------- Generate Embedding ----------

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorise *texts* → normalised float32 numpy array."""
        return self.embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    # ---------- Predict Positive Probability ----------

    def _predict_pos(
        self,
        query_text: str,
        cand_texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> List[ScoreT]:
        """Return P(positive) for each (query, cand) pair in *cand_texts*."""
        probs: List[ScoreT] = []
        for i in range(0, len(cand_texts), batch_size):
            chunk = cand_texts[i: i + batch_size]
            encoding = self.clf_tokenizer(
                [query_text] * len(chunk),
                chunk,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.clf_model(**encoding).logits
                chunk_probs = F.softmax(logits, dim=1)[:, self.POS_CLASS_IDX]
                probs.extend(chunk_probs.cpu().tolist())
        return probs

    # ---------- Stage 1: Retrieval ----------

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

        # Compute cosine similarity matrix (normalised vectors)
        similarity_matrix = np.matmul(query_embeddings, source_embeddings.T)

        # Keep only the top-k most similar segments for each query segment
        similarity_results: SimDict = {}
        for qi, qseg in enumerate(query_segments):
            ranked_idx = (
                np.argsort(similarity_matrix[qi])[::-1][:top_k]
                if top_k < similarity_matrix.shape[1]
                else np.argsort(similarity_matrix[qi])[::-1]
            )
            similarity_results[qseg.id] = [
                (source_segments[si], float(similarity_matrix[qi, si]))
                for si in ranked_idx
            ]

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
        if candidates is None:
            candidates = self.generate_candidates(
                query=query,
                source=source,
                **kwargs,
            )

        full_results: FullDict = {}
        for qid, sim_pairs in candidates.items():
            cand_texts = [seg.text for seg, _ in sim_pairs]
            probabilities = self._predict_pos(
                query[qid].text, cand_texts, batch_size=batch_size
            )
            full_results[qid] = [
                (seg, sim_score, prob_pos)
                for (seg, sim_score), prob_pos in zip(sim_pairs, probabilities)
            ]

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
        embedding_model_name="bowphs/PhilBerta",
        device="cpu",
    )
    
    # Run the pipeline with the query and source documents
    results = pipeline.run(
        query=query_doc, # Query document
        source=source_doc, # Source document
        top_k=3, # Number of top similar candidates to classify
    )
    pretty_print(results)
