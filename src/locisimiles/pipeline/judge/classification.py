# pipeline/judge/classification.py
"""Classification judge using a transformer sequence-classification model."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from locisimiles.document import Document
from locisimiles.pipeline._types import (
    CandidateJudge,
    CandidateGeneratorOutput,
    CandidateJudgeOutput,
)
from locisimiles.pipeline.judge._base import JudgeBase


class ClassificationJudge(JudgeBase):
    """Judge candidates using a transformer classification model.

    Loads a pre-trained sequence-classification model and tokenizer.
    For each query–candidate pair the model outputs P(positive), which
    is stored as ``judgment_score``.

    Args:
        classification_name: HuggingFace model identifier.
        device: Torch device string.
        pos_class_idx: Index of the positive class in the classifier output.

    Example:
        ```python
        from locisimiles.pipeline.judge import ClassificationJudge
        from locisimiles.document import Document

        judge = ClassificationJudge(device="cpu")
        results = judge.judge(query=query_doc, candidates=candidates)
        ```
    """

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/PhilBerta-class-latin-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,
    ):
        self.device = device if device is not None else "cpu"
        self.pos_class_idx = pos_class_idx

        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(
            classification_name
        )
        self.clf_model.to(self.device).eval()

    # ---------- Tokenizer helpers ----------

    def _count_special_tokens_added(self) -> int:
        """Count special tokens added by the tokenizer for pair encoding."""
        return self.clf_tokenizer.num_special_tokens_to_add(pair=True)

    def _truncate_pair(
        self,
        sentence1: str,
        sentence2: str,
        max_len: int = 512,
    ) -> Tuple[str, str]:
        """Truncate a text pair to fit within *max_len* including specials."""
        num_special = self._count_special_tokens_added()
        max_tokens = max_len - num_special
        half = max_tokens // 2

        tokens1 = self.clf_tokenizer.tokenize(sentence1)[:half]
        tokens2 = self.clf_tokenizer.tokenize(sentence2)[:half]

        sentence1 = self.clf_tokenizer.convert_tokens_to_string(tokens1)
        sentence2 = self.clf_tokenizer.convert_tokens_to_string(tokens2)
        return sentence1, sentence2

    # ---------- Prediction ----------

    def _predict_batch(
        self,
        query_text: str,
        cand_texts: Sequence[str],
        max_len: int = 512,
    ) -> List[float]:
        """Predict P(positive) for a batch of (query, candidate) pairs."""
        truncated_pairs = [
            self._truncate_pair(query_text, ct, max_len) for ct in cand_texts
        ]
        query_texts_trunc = [p[0] for p in truncated_pairs]
        cand_texts_trunc = [p[1] for p in truncated_pairs]

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
    ) -> List[float]:
        """Return P(positive) for each (query, cand) pair, with internal batching."""
        probs: List[float] = []
        for i in range(0, len(cand_texts), batch_size):
            chunk = cand_texts[i : i + batch_size]
            probs.extend(self._predict_batch(query_text, chunk, max_len=max_len))
        return probs

    def debug_input_sequence(
        self,
        query_text: str,
        candidate_text: str,
        max_len: int = 512,
    ) -> Dict[str, Any]:
        """Inspect how a query–candidate pair is tokenised and encoded.

        Returns a dictionary with original / truncated texts, token IDs,
        attention mask, and decoded input text with special tokens visible.
        """
        query_trunc, candidate_trunc = self._truncate_pair(
            query_text, candidate_text, max_len
        )
        encoding = self.clf_tokenizer(
            query_trunc,
            candidate_trunc,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        decoded_text = self.clf_tokenizer.decode(
            encoding["input_ids"].squeeze(), skip_special_tokens=False
        )
        return {
            "query": query_text,
            "candidate": candidate_text,
            "query_truncated": query_trunc,
            "candidate_truncated": candidate_trunc,
            "input_ids": encoding["input_ids"].squeeze().tolist(),
            "attention_mask": encoding["attention_mask"].squeeze().tolist(),
            "input_text": decoded_text,
        }

    # ---------- JudgeBase ----------

    def judge(
        self,
        *,
        query: Document,
        candidates: CandidateGeneratorOutput,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> CandidateJudgeOutput:
        """Classify each candidate pair using the loaded model.

        Args:
            query: Query document.
            candidates: Output from a candidate generator.
            batch_size: Batch size for the classifier.

        Returns:
            ``CandidateJudgeOutput`` with ``judgment_score`` =
            P(positive) from the classifier.
        """
        judge_results: CandidateJudgeOutput = {}

        for query_id, candidate_list in tqdm(
            candidates.items(), desc="Judging candidates"
        ):
            cand_texts = [c.segment.text for c in candidate_list]
            probabilities = self._predict(
                query[query_id].text,
                cand_texts,
                batch_size=batch_size,
            )

            judge_results[query_id] = [
                CandidateJudge(
                    segment=candidate.segment,
                    candidate_score=candidate.score,
                    judgment_score=probability,
                )
                for candidate, probability in zip(candidate_list, probabilities)
            ]

        return judge_results
