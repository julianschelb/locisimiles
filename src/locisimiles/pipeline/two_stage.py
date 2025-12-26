# pipeline/two_stage.py
"""
Two-stage pipeline: Retrieval (candidate generation) + Classification.
"""
from __future__ import annotations

import time
import chromadb
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Sequence
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from locisimiles.document import Document, TextSegment
from locisimiles.pipeline._types import ScoreT, SimDict, FullDict
from tqdm import tqdm


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
        classification_name: str = "julian-schelb/PhilBerta-class-latin-intertext-v1",
        embedding_model_name: str = "julian-schelb/SPhilBerta-emb-lat-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,  # Index of the positive class in the classifier
    ):
        self.device = device if device is not None else "cpu"
        self.pos_class_idx = pos_class_idx
        self._source_index: chromadb.Collection | None = None

        # -------- Load Models ----------
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classification_name)
        self.clf_model.to(self.device).eval()

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

    # ---------- Predict Positive Probability ----------

    def _count_special_tokens_added(self) -> int:
        """Counts the number of special tokens added by the tokenizer."""
        return self.clf_tokenizer.num_special_tokens_to_add(pair=True)
    
    def _truncate_pair(self, sentence1: str, sentence2: str, max_len: int = 512) -> Tuple[str, str]:
        """Truncates sentence1 and sentence2 to fit within max_len including special tokens."""
        num_special = self._count_special_tokens_added()
        max_tokens = max_len - num_special
        half = max_tokens // 2

        # Tokenize and truncate
        tokens1 = self.clf_tokenizer.tokenize(sentence1)[:half]
        tokens2 = self.clf_tokenizer.tokenize(sentence2)[:half]

        # Convert back to string
        sentence1 = self.clf_tokenizer.convert_tokens_to_string(tokens1)
        sentence2 = self.clf_tokenizer.convert_tokens_to_string(tokens2)
        return sentence1, sentence2

    def _predict_batch(
        self,
        query_text: str,
        cand_texts: Sequence[str],
        max_len: int = 512,
    ) -> List[ScoreT]:
        """Predict probabilities for a batch of (query, cand) pairs."""
        # Truncate pairs intelligently before tokenization
        truncated_pairs = [self._truncate_pair(query_text, cand_text, max_len) 
                          for cand_text in cand_texts]
        query_texts_trunc = [pair[0] for pair in truncated_pairs]
        cand_texts_trunc = [pair[1] for pair in truncated_pairs]
        
        encoding = self.clf_tokenizer(
            query_texts_trunc,
            cand_texts_trunc,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
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

    def debug_input_sequence(self, query_text: str, candidate_text: str, max_len: int = 512) -> Dict[str, Any]:
        """Debug method to inspect how a query-candidate pair is encoded.
        
        Returns a dictionary with:
        - query: Original query text
        - candidate: Original candidate text
        - query_truncated: Truncated query text
        - candidate_truncated: Truncated candidate text
        - input_ids: Token IDs as list
        - input_text: Decoded text with special tokens visible
        - attention_mask: Attention mask as list
        """
        # Truncate the pair
        query_trunc, candidate_trunc = self._truncate_pair(query_text, candidate_text, max_len)
        
        # Encode the pair
        encoding = self.clf_tokenizer(
            query_trunc,
            candidate_trunc,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        
        # Decode with special tokens visible
        decoded_text = self.clf_tokenizer.decode(encoding['input_ids'].squeeze(), skip_special_tokens=False)
        
        return {
            "query": query_text,
            "candidate": candidate_text,
            "query_truncated": query_trunc,
            "candidate_truncated": candidate_trunc,
            "input_ids": encoding['input_ids'].squeeze().tolist(),
            "attention_mask": encoding['attention_mask'].squeeze().tolist(),
            "input_text": decoded_text,
        }

    # ---------- Stage 1: Retrieval ----------
    
    def build_source_index(
        self,
        source_segments: Sequence[TextSegment],
        source_embeddings: np.ndarray,
        collection_name: str = "source_segments",
        batch_size: int = 5000,  # Safe batch size
    ):
        """Create a Chroma collection from *source_segments* and their embeddings."""
        
        # Use EphemeralClient for non-persistent, in-memory storage
        # Create new client each time to ensure clean state
        client = chromadb.EphemeralClient()
        
        # Use unique collection name to avoid conflicts in same session
        unique_name = f"{collection_name}_{int(time.time() * 1000000)}"
        
        # Delete collection if it exists (should not happen with unique names, but just in case)
        try:
            client.delete_collection(name=unique_name)
        except Exception:
            pass  # Collection doesn't exist, which is fine
        
        # Create fresh collection with unique name
        col = client.create_collection(
            name=unique_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance
        )

        # Extract IDs and embeddings
        ids = [s.id for s in source_segments]
        embeddings = source_embeddings.tolist()

        # Add segments to the collection in batches
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

        # Iterate over each query segment and its embedding
        for query_segment, query_embedding in zip(query_segments, query_embeddings):
            
            # Query the Chroma index for the top-k similar source segments
            results = self._source_index.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
            
            # Map the results to TextSegment objects and similarity scores
            # Convert cosine distance to cosine similarity: similarity = 1 - distance
            similarity_results[query_segment.id] = [
                (source_document[idx], 1.0 - float(distance))
                for idx, distance in zip(results["ids"][0], results["distances"][0])
            ]

        return similarity_results

    def generate_candidates(
        self,
        *,
        query: Document,
        source: Document,
        top_k: int = 5,
        query_prompt_name: str = "query",
        source_prompt_name: str = "match",
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
        query_embeddings = self._embed(
            [s.text for s in tqdm(query_segments, desc="Embedding query segments")],
            prompt_name=query_prompt_name
        )
        
        # Embed source segments with a progress bar
        source_embeddings = self._embed(
            [s.text for s in tqdm(source_segments, desc="Embedding source segments")],
            prompt_name=source_prompt_name
        )
        
        # Build the source index for fast retrieval
        self._source_index = self.build_source_index(
            source_segments=source_segments,
            source_embeddings=source_embeddings,
            collection_name="source_segments",
        )
        
        # Compute similarity between query and source segments
        similarity_results = self._compute_similarity(
            query_segments=query_segments,
            query_embeddings=query_embeddings,
            source_document=source,
            top_k=top_k,
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
        
        for query_id, similarity_pairs in tqdm(candidates.items(), desc="Check candidates"):
            
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
        query_prompt_name: str = "query",
        source_prompt_name: str = "match",
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
            query_prompt_name=query_prompt_name,
            source_prompt_name=source_prompt_name,
        )
        return self.check_candidates(
            query=query,
            source=source,
            candidates=similarity_dict,
            **kwargs,
        )
