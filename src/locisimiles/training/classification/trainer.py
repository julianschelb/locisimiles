# training/classification/trainer.py
"""Trainer for the fine-tuned transformer sequence classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from locisimiles.training.artifacts import resolve_model_output_path
from locisimiles.training.base import BaseTrainer, TrainerConfig
from locisimiles.training.classification.threshold import ThresholdSet
from locisimiles.training.classification.threshold import tune_threshold as _tune_threshold
from locisimiles.training.classification.tokenizer_utils import (
    add_roberta_separators,
    truncate_pair,
)
from locisimiles.training.data import TrainingData


@dataclass(frozen=True)
class ClassificationTrainerConfig(TrainerConfig):
    """Configuration for the transformer sequence-classification trainer.

    ``fit()`` takes a ``TrainingData`` whose ground truth ``label`` may be
    binary (e.g. ``no_match``/``match``) or multiclass (e.g.
    ``no_match``/``cit``/``cf``) — the number of classes is inferred from the
    distinct labels seen during ``fit()``.
    """

    model_name: str = "xlm-roberta-base"
    label_names: Optional[Dict[int, str]] = None
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 512
    class_weight: Optional[Literal["balanced"]] = None
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    apply_roberta_separator_fix: bool = False
    disable_compile: bool = False
    device: str = "cpu"
    output_filename: str = "classifier"


class ClassificationTrainer(BaseTrainer):
    """Fine-tune a transformer sequence-classification model on labeled pairs.

    Mirrors the experiment recipe: a hand-rolled ``AdamW`` loop, no LR
    scheduler, a fixed epoch count (no early stopping/checkpoint selection —
    the model is saved after the last epoch), with optional balanced
    class-weighting or focal loss. Pair truncation reuses the exact strategy
    :class:`~locisimiles.pipeline.judge.classification.ClassificationJudge`
    uses at inference time, so train/inference tokenization matches.
    """

    def __init__(self, config: ClassificationTrainerConfig):
        super().__init__(config)
        self.model: Any = None
        self.tokenizer: Any = None
        self._label_to_id: Optional[Dict[str, int]] = None
        self.threshold_set: Optional[ThresholdSet] = None

    @property
    def cfg(self) -> ClassificationTrainerConfig:
        return self.config  # type: ignore[return-value]

    def validate_data(self) -> None:
        """Ensure the output directory exists; ``fit()`` validates its ``TrainingData``."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_labels(data: TrainingData) -> Dict[str, int]:
        labels = sorted({str(label) for _, _, label in data})
        return {label: idx for idx, label in enumerate(labels)}

    def _build_classifier(self, num_labels: int) -> Any:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        if self.cfg.apply_roberta_separator_fix:
            add_roberta_separators(self.tokenizer)
        if self.cfg.disable_compile:
            model_config = getattr(self.model, "config", None)
            if model_config is not None and hasattr(model_config, "reference_compile"):
                model_config.reference_compile = False
        self.model.to(self.cfg.device)
        return self.model

    def _class_weights(self, label_ids: List[int], num_labels: int) -> Any:
        import torch

        counts = [0] * num_labels
        for label_id in label_ids:
            counts[label_id] += 1
        n = len(label_ids)
        weights = [n / (num_labels * count) if count > 0 else 0.0 for count in counts]
        return torch.tensor(weights, dtype=torch.float32, device=self.cfg.device)

    def fit(self, *, data: TrainingData, **kwargs: Any) -> Any:  # type: ignore[override]
        """Fine-tune the classifier on resolved ``(query_text, source_text, label)`` pairs."""
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset

        self.validate_data()
        torch.manual_seed(self.cfg.seed)

        rows = list(data)
        if not rows:
            raise ValueError("No training rows found in TrainingData")

        self._label_to_id = self._resolve_labels(data)
        label_to_id = self._label_to_id
        num_labels = len(label_to_id)
        self._build_classifier(num_labels)
        tokenizer = self.tokenizer
        max_len = self.cfg.max_length

        class _PairDataset(Dataset):
            def __init__(self_inner, rows: List[Tuple[str, str, Any]]):
                self_inner.rows = rows

            def __len__(self_inner) -> int:
                return len(self_inner.rows)

            def __getitem__(self_inner, idx: int) -> Tuple[str, str, int]:
                query_text, source_text, label = self_inner.rows[idx]
                sentence1, sentence2 = truncate_pair(tokenizer, query_text, source_text, max_len)
                return sentence1, sentence2, label_to_id[str(label)]

        def _collate(batch: List[Tuple[str, str, int]]) -> Tuple[Any, Any]:
            sentence1s = [item[0] for item in batch]
            sentence2s = [item[1] for item in batch]
            labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
            encoding = tokenizer(
                sentence1s,
                sentence2s,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            return encoding, labels

        loader = DataLoader(
            _PairDataset(rows),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=_collate,
        )

        class_weights = None
        if self.cfg.class_weight == "balanced":
            label_ids = [label_to_id[str(label)] for _, _, label in rows]
            class_weights = self._class_weights(label_ids, num_labels)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        self.model.train()
        for _epoch in range(self.cfg.epochs):
            for encoding, labels in loader:
                encoding = {key: value.to(self.cfg.device) for key, value in encoding.items()}
                labels = labels.to(self.cfg.device)

                optimizer.zero_grad()
                logits = self.model(**encoding).logits

                if self.cfg.use_focal_loss:
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                    ce = F.nll_loss(log_probs, labels, weight=class_weights, reduction="none")
                    loss = ((1 - pt) ** self.cfg.focal_gamma * ce).mean()
                elif class_weights is not None:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
                else:
                    loss = F.cross_entropy(logits, labels)

                loss.backward()
                optimizer.step()

        self.model.eval()
        return self.model

    def _predict_probabilities(self, data: TrainingData) -> Tuple[List[List[float]], List[str]]:
        import torch
        import torch.nn.functional as F

        if self.model is None or self.tokenizer is None or self._label_to_id is None:
            raise ValueError("No trained model available. Call fit() first.")

        rows = list(data)
        all_probs: List[List[float]] = []
        gold_labels: List[str] = []
        batch_size = self.cfg.batch_size

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                sentence1s, sentence2s, labels = [], [], []
                for query_text, source_text, label in chunk:
                    s1, s2 = truncate_pair(
                        self.tokenizer, query_text, source_text, self.cfg.max_length
                    )
                    sentence1s.append(s1)
                    sentence2s.append(s2)
                    labels.append(str(label))
                encoding = self.tokenizer(
                    sentence1s,
                    sentence2s,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                ).to(self.cfg.device)
                logits = self.model(**encoding).logits
                all_probs.extend(F.softmax(logits, dim=1).cpu().tolist())
                gold_labels.extend(labels)

        return all_probs, gold_labels

    def tune_threshold(
        self,
        *,
        data: TrainingData,
        method: str = "max_f1",
        negative_label: str = "no_match",
        **kwargs: Any,
    ) -> ThresholdSet:
        """Tune one-vs-rest decision thresholds on an evaluation ``TrainingData``.

        For multiclass label schemes, tunes an independent threshold per
        positive class; ties between classes that both clear their threshold
        are broken by comparing probabilities (the only tie-break rule
        currently supported).
        """
        if self._label_to_id is None:
            raise ValueError("No trained model available. Call fit() first.")
        id_to_label = {idx: label for label, idx in self._label_to_id.items()}
        probabilities, gold_labels = self._predict_probabilities(data)
        self.threshold_set = _tune_threshold(
            probabilities=probabilities,
            gold_labels=gold_labels,
            id_to_label=id_to_label,
            method=method,
            negative_label=negative_label,
        )
        return self.threshold_set

    def save(self, **kwargs: Any) -> Path:
        """Persist the fine-tuned model, tokenizer, and (if tuned) thresholds.

        Sets ``model.config.id2label``/``label2id`` from the resolved label
        mapping before saving, so the saved directory is immediately usable
        by :class:`~locisimiles.pipeline.judge.classification.ClassificationJudge`
        with no manual step.
        """
        if self.model is None or self.tokenizer is None or self._label_to_id is None:
            raise ValueError("No trained model available. Call fit() first.")

        label_names = self.cfg.label_names or {
            idx: label for label, idx in self._label_to_id.items()
        }
        self.model.config.id2label = {int(idx): str(name) for idx, name in label_names.items()}
        self.model.config.label2id = {str(name): int(idx) for idx, name in label_names.items()}

        output_path = resolve_model_output_path(self.cfg.output_dir, self.cfg.output_filename)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        if self.threshold_set is not None:
            self.threshold_set.to_json(output_path / "threshold.json")

        return output_path

    def load_artifacts(self, path: str | Path) -> Any:
        """Load a previously saved classifier directory."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.cfg.device).eval()

        id2label = getattr(self.model.config, "id2label", None) or {}
        self._label_to_id = {str(label): int(idx) for idx, label in id2label.items()}
        return self.model
