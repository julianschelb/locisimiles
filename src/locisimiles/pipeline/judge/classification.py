# pipeline/judge/classification.py
"""Classification judge using a transformer sequence-classification model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from locisimiles.document import Document
from locisimiles.pipeline._types import (
    CandidateGeneratorOutput,
    CandidateJudge,
    CandidateJudgeOutput,
)
from locisimiles.pipeline.judge._base import CandidateJudgeBase

DEFAULT_NEGATIVE_LABELS = {
    "0",
    "label_0",
    "negative",
    "no",
    "none",
    "no_match",
    "no-match",
    "no match",
    "not_intertext",
    "non_link",
    "non-link",
}


@dataclass(frozen=True)
class _ClassificationPrediction:
    """Internal representation of one classifier output row."""

    judgment_score: float
    predicted_class_id: int
    predicted_label: str
    class_probabilities: Dict[str, float]


def _normalise_label(label: object) -> str:
    """Canonicalise class labels for matching user/model metadata."""
    return str(label).strip().lower().replace("-", "_").replace(" ", "_").rstrip(".")


class ClassificationJudge(CandidateJudgeBase):
    """Judge candidates using a transformer classification model.

    Loads a pre-trained sequence-classification model and tokenizer.
    For each query-candidate pair the model stores a link score in
    ``judgment_score``.  For binary models this is the configured positive
    class probability.  For multiclass models it is the summed probability
    of the configured positive classes.  The argmax class and full class
    probability distribution are also stored on each ``CandidateJudge``.

    The default model is
    ``julian-schelb/xlm-roberta-large-class-lat-intertext-v1``, a fine-tuned
    classifier for Latin intertextuality detection.

    Args:
        classification_name: HuggingFace model identifier.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        pos_class_idx: Index of the positive class in the classifier output.
            Kept for binary models and as a fallback when no positive classes
            can be inferred.
        label_names: Optional class label mapping.  Pass either a sequence
            ordered by class id or a mapping from class id to label.  This is
            useful when a model config contains generic labels like
            ``LABEL_0``.
        positive_class_ids: Optional class ids whose probabilities are summed
            into ``judgment_score``.
        positive_labels: Optional class labels whose probabilities are summed
            into ``judgment_score``.
        negative_labels: Optional class labels treated as non-links.  When no
            positive classes are provided, all non-negative classes are treated
            as positive for multiclass models.
        emit_class_metadata: Whether to attach predicted labels and class
            probabilities to output results.  Defaults to automatic behavior:
            enabled for multiclass or explicitly label-configured models,
            disabled for default binary models.

    Example:
        ```python
        from locisimiles.pipeline.judge import ClassificationJudge

        # Create judge with default model
        judge = ClassificationJudge(device="cpu")

        # Score pre-generated candidates
        results = judge.judge(query=query_doc, candidates=candidates)

        # Each result has a judgment_score (probability of being a match)
        for qid, judgments in results.items():
            for j in judgments:
                if j.judgment_score > 0.5:
                    print(f"{qid} → {j.segment.id}: {j.judgment_score:.3f}")
        ```
    """

    def __init__(
        self,
        *,
        classification_name: str = "julian-schelb/xlm-roberta-large-class-lat-intertext-v1",
        device: str | int | None = None,
        pos_class_idx: int = 1,
        label_names: Sequence[str] | Mapping[int | str, str] | None = None,
        positive_class_ids: Sequence[int] | None = None,
        positive_labels: Sequence[str] | None = None,
        negative_labels: Sequence[str] | None = None,
        emit_class_metadata: bool | None = None,
    ):
        self.device = device if device is not None else "cpu"
        self.pos_class_idx = pos_class_idx

        self.clf_tokenizer = AutoTokenizer.from_pretrained(classification_name)
        self.clf_model = AutoModelForSequenceClassification.from_pretrained(classification_name)
        self.clf_model.to(self.device).eval()

        self.label_names = self._resolve_label_names(label_names)
        self.positive_class_ids = list(positive_class_ids) if positive_class_ids is not None else None
        self.positive_labels = (
            {_normalise_label(label) for label in positive_labels}
            if positive_labels is not None
            else None
        )
        self.negative_labels = DEFAULT_NEGATIVE_LABELS | {
            _normalise_label(label) for label in (negative_labels or [])
        }
        self.num_labels = max(self.label_names) + 1 if self.label_names else 2
        has_explicit_class_config = any(
            value is not None
            for value in (label_names, positive_class_ids, positive_labels, negative_labels)
        )
        self.emit_class_metadata = (
            self.num_labels > 2 or has_explicit_class_config
            if emit_class_metadata is None
            else emit_class_metadata
        )

    # ---------- Label helpers ----------

    def _resolve_label_names(
        self,
        label_names: Sequence[str] | Mapping[int | str, str] | None,
    ) -> Dict[int, str]:
        """Resolve class-id labels from explicit input or model config."""
        if label_names is not None:
            if isinstance(label_names, Mapping):
                resolved: Dict[int, str] = {}
                for raw_idx, label in label_names.items():
                    resolved[int(raw_idx)] = str(label)
                return resolved
            return {idx: str(label) for idx, label in enumerate(label_names)}

        config = getattr(self.clf_model, "config", None)
        id2label = getattr(config, "id2label", None)
        if isinstance(id2label, Mapping):
            resolved = {}
            for raw_idx, label in id2label.items():
                try:
                    resolved[int(raw_idx)] = str(label)
                except (TypeError, ValueError):
                    continue
            if resolved:
                return resolved

        num_labels = getattr(config, "num_labels", None)
        if not isinstance(num_labels, int) or num_labels <= 0:
            num_labels = 2
        return {idx: f"LABEL_{idx}" for idx in range(num_labels)}

    def _label_for_class_id(self, class_id: int) -> str:
        """Return a stable label for a class id."""
        return self.label_names.get(class_id, f"LABEL_{class_id}")

    def _positive_class_ids(self, num_labels: int) -> List[int]:
        """Resolve class ids that count as positive intertextual links."""
        if self.positive_class_ids is not None:
            return [idx for idx in self.positive_class_ids if 0 <= idx < num_labels]

        if self.positive_labels is not None:
            return [
                idx
                for idx in range(num_labels)
                if _normalise_label(self._label_for_class_id(idx)) in self.positive_labels
            ]

        if num_labels <= 2:
            return [self.pos_class_idx] if 0 <= self.pos_class_idx < num_labels else []

        return [
            idx
            for idx in range(num_labels)
            if _normalise_label(self._label_for_class_id(idx)) not in self.negative_labels
        ]

    def _prediction_from_probabilities(self, probabilities: Sequence[float]) -> _ClassificationPrediction:
        """Build classifier metadata from one softmax probability row."""
        prob_list = [float(probability) for probability in probabilities]
        predicted_class_id = max(range(len(prob_list)), key=prob_list.__getitem__)
        positive_ids = self._positive_class_ids(len(prob_list))
        judgment_score = sum(prob_list[idx] for idx in positive_ids)
        class_probabilities = {
            self._label_for_class_id(idx): probability for idx, probability in enumerate(prob_list)
        }
        return _ClassificationPrediction(
            judgment_score=judgment_score,
            predicted_class_id=predicted_class_id,
            predicted_label=self._label_for_class_id(predicted_class_id),
            class_probabilities=class_probabilities,
        )

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
        return [
            prediction.judgment_score
            for prediction in self._predict_batch_details(query_text, cand_texts, max_len=max_len)
        ]

    def _predict_batch_details(
        self,
        query_text: str,
        cand_texts: Sequence[str],
        max_len: int = 512,
    ) -> List[_ClassificationPrediction]:
        """Predict full class metadata for a batch of (query, candidate) pairs."""
        truncated_pairs = [self._truncate_pair(query_text, ct, max_len) for ct in cand_texts]
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
            probabilities = F.softmax(logits, dim=1).cpu().tolist()
            return [self._prediction_from_probabilities(row) for row in probabilities]

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

    def _predict_details(
        self,
        query_text: str,
        cand_texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_len: int = 512,
    ) -> List[_ClassificationPrediction]:
        """Return full class metadata for each pair, with internal batching."""
        predictions: List[_ClassificationPrediction] = []
        for i in range(0, len(cand_texts), batch_size):
            chunk = cand_texts[i : i + batch_size]
            predictions.extend(self._predict_batch_details(query_text, chunk, max_len=max_len))
        return predictions

    def debug_input_sequence(
        self,
        query_text: str,
        candidate_text: str,
        max_len: int = 512,
    ) -> Dict[str, Any]:
        """Inspect how a query–candidate pair is tokenised and encoded.

        Useful for debugging classification results or understanding
        how text truncation affects model input.

        Args:
            query_text: Raw query text.
            candidate_text: Raw candidate text.
            max_len: Maximum token length.

        Returns:
            Dictionary with keys:

            - ``query`` / ``candidate`` — original texts.
            - ``query_truncated`` / ``candidate_truncated`` — after truncation.
            - ``input_ids`` — token ID list.
            - ``attention_mask`` — attention mask list.
            - ``input_text`` — decoded input with special tokens visible.

        Example:
            ```python
            judge = ClassificationJudge(device="cpu")
            info = judge.debug_input_sequence(
                "Arma virumque cano",
                "Troiae qui primus ab oris",
            )
            print(info["input_text"])
            ```
        """
        query_trunc, candidate_trunc = self._truncate_pair(query_text, candidate_text, max_len)
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

        for query_id, candidate_list in tqdm(candidates.items(), desc="Judging candidates"):
            cand_texts = [c.segment.text for c in candidate_list]
            predictions = self._predict_details(
                query[query_id].text,
                cand_texts,
                batch_size=batch_size,
            )

            judgments = []
            for candidate, prediction in zip(candidate_list, predictions):
                if self.emit_class_metadata:
                    judgments.append(
                        CandidateJudge(
                            segment=candidate.segment,
                            candidate_score=candidate.score,
                            judgment_score=prediction.judgment_score,
                            predicted_class_id=prediction.predicted_class_id,
                            predicted_label=prediction.predicted_label,
                            class_probabilities=prediction.class_probabilities,
                        )
                    )
                else:
                    judgments.append(
                        CandidateJudge(
                            segment=candidate.segment,
                            candidate_score=candidate.score,
                            judgment_score=prediction.judgment_score,
                        )
                    )
            judge_results[query_id] = judgments

        return judge_results
