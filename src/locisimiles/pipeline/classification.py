# pipeline/classification.py
"""
Classification-only pipeline: Exhaustive pairwise comparison without retrieval.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from locisimiles.document import Document
from locisimiles.pipeline._types import ScoreT, FullDict
from tqdm import tqdm


class ClassificationPipeline:
    """
    A simpler pipeline for intertextuality classification without candidate generation.
    It classifies all possible pairs between query and source segments without
    a retrieval stage. Suitable for smaller document pairs or when exhaustive
    comparison is needed.
    """
    
    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/PhilBerta-class-latin-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,  # Index of the positive class in the classifier
    ):
        self.device = device if device is not None else "cpu"
        self.pos_class_idx = pos_class_idx

        # -------- Load Classification Model ----------
        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classification_name)
        self.clf_model.to(self.device).eval()

        # Keep results in memory for later access
        self._last_results: FullDict | None = None

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
        max_len: int = 512,
    ) -> List[ScoreT]:
        """Return P(positive) for each (query, cand) pair in *cand_texts*."""
        probs: List[ScoreT] = []
        
        # Predict in batches between a query and multiple candidates
        for i in range(0, len(cand_texts), batch_size):
            chunk = cand_texts[i: i + batch_size]
            chunk_probs = self._predict_batch(query_text, chunk, max_len=max_len)
            probs.extend(chunk_probs)
        return probs

    def debug_input_sequence(self, query_text: str, candidate_text: str, max_len: int = 512) -> Dict[str, Any]:
        """Debug method to inspect how a query-candidate pair is encoded."""
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

    # ---------- Main Pipeline ----------

    def run(
        self,
        *,
        query: Document,
        source: Document,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> FullDict:
        """
        Run classification on all query-source segment pairs.
        Returns a dictionary mapping query segment IDs to lists of
        (source segment, similarity_score=None, P(positive)) tuples.
        
        Note: Since there's no retrieval stage, similarity_score is set to None.
        """
        results: FullDict = {}
        
        # Extract all source segments
        source_segments = list(source.segments.values())
        source_texts = [s.text for s in source_segments]
        
        # For each query segment, classify against all source segments
        for query_segment in tqdm(query.segments.values(), desc="Classifying pairs"):
            query_text = query_segment.text
            
            # Predict probabilities for all source segments
            probabilities = self._predict(
                query_text, 
                source_texts, 
                batch_size=batch_size
            )
            
            # Build results with None for similarity score (no retrieval stage)
            results[query_segment.id] = [
                (source_seg, None, prob)
                for source_seg, prob in zip(source_segments, probabilities)
            ]
        
        self._last_results = results
        return results
