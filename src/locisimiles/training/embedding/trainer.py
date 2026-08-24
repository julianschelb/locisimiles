# training/embedding/trainer.py
"""Trainer for the fine-tuned SentenceTransformer bi-encoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from locisimiles.training.artifacts import resolve_model_output_path
from locisimiles.training.base import BaseTrainer, TrainerConfig
from locisimiles.training.data import TrainingData


def _default_prompts() -> Dict[str, str]:
    return {"query": "query: ", "match": "passage: "}


@dataclass(frozen=True)
class EmbeddingTrainerConfig(TrainerConfig):
    """Configuration for the SentenceTransformer bi-encoder trainer.

    ``prompts`` maps the dataset column names ``fit()`` builds internally
    (``"query"``/``"match"``) to the asymmetric prefixes prepended at
    training/inference time (E5-style by default). This is exactly what
    :class:`~locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator`
    later calls with ``prompt_name="query"``/``"match"``.
    """

    model_name: str = "intfloat/multilingual-e5-small"
    loss_type: Literal["online_contrastive", "contrastive"] = "online_contrastive"
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    prompts: Dict[str, str] = field(default_factory=_default_prompts)
    negative_label: str = "no_match"
    device: str = "cpu"
    output_filename: str = "embedding_model"
    #: Select the epoch with the best ``eval_data`` score (average precision
    #: over cosine similarity) instead of always keeping the last epoch.
    #: Requires ``eval_data`` at ``fit()``. Off by default, matching the
    #: paper's fixed-epoch-then-save recipe.
    select_best_checkpoint: bool = False
    #: Stop training early if the ``eval_data`` score hasn't improved for
    #: this many evaluations. Requires ``select_best_checkpoint=True``.
    early_stopping_patience: Optional[int] = None


class EmbeddingTrainer(BaseTrainer):
    """Fine-tune a SentenceTransformer bi-encoder on labeled query/source pairs.

    Uses ``OnlineContrastiveLoss`` (the paper's production loss) by default;
    triplet-loss variants tried in the experiments were superseded and are
    not ported. By default, trains for a fixed epoch count with no
    checkpoint selection, matching the experiment recipe; pass
    ``select_best_checkpoint=True`` (and ``eval_data``) to opt into
    best-epoch selection and, optionally, early stopping.
    """

    def __init__(self, config: EmbeddingTrainerConfig):
        super().__init__(config)
        self.model: Any = None

    @property
    def cfg(self) -> EmbeddingTrainerConfig:
        return self.config  # type: ignore[return-value]

    def validate_data(self) -> None:
        """Ensure the output directory exists; ``fit()`` validates its ``TrainingData``."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_dataset(self, data: TrainingData) -> Any:
        from datasets import Dataset

        query_key, match_key = "query", "match"
        rows: Dict[str, list] = {query_key: [], match_key: [], "label": []}
        for query_text, source_text, label in data:
            rows[query_key].append(query_text)
            rows[match_key].append(source_text)
            rows["label"].append(0.0 if str(label) == self.cfg.negative_label else 1.0)
        return Dataset.from_dict(rows)

    def _build_loss(self, model: Any) -> Any:
        if self.cfg.loss_type == "online_contrastive":
            from sentence_transformers.losses import OnlineContrastiveLoss

            return OnlineContrastiveLoss(model)
        if self.cfg.loss_type == "contrastive":
            from sentence_transformers.losses import ContrastiveLoss

            return ContrastiveLoss(model)
        raise ValueError(f"Unknown loss_type: {self.cfg.loss_type!r}")

    def fit(  # type: ignore[override]
        self,
        *,
        data: TrainingData,
        eval_data: Optional[TrainingData] = None,
        **kwargs: Any,
    ) -> Any:
        """Fine-tune the embedding model on resolved ``(query_text, source_text, label)`` pairs.

        ``eval_data``, if given, runs a ``BinaryClassificationEvaluator`` once
        per epoch. By default (``select_best_checkpoint=False``) this is
        purely for visibility — the final epoch's weights are what get
        returned/saved, matching the paper's fixed-epoch-then-save recipe. Set
        ``select_best_checkpoint=True`` to instead keep the epoch with the
        best ``eval_data`` score (average precision over cosine similarity),
        optionally with ``early_stopping_patience`` to stop early once that
        score stops improving.
        """
        import torch
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )

        if self.cfg.select_best_checkpoint and eval_data is None:
            raise ValueError("select_best_checkpoint=True requires eval_data")
        if self.cfg.early_stopping_patience is not None and not self.cfg.select_best_checkpoint:
            raise ValueError("early_stopping_patience requires select_best_checkpoint=True")

        self.validate_data()
        torch.manual_seed(self.cfg.seed)

        train_dataset = self._build_dataset(data)
        if len(train_dataset) == 0:
            raise ValueError("No training rows found in TrainingData")

        self.model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
        # `SentenceTransformerTrainingArguments(prompts=...)` below only controls
        # how prompts are *applied to training inputs*; it does not persist onto
        # the model itself. Set it directly so the saved model's own `prompts`
        # (what `SentenceTransformer.encode(..., prompt_name=...)` reads at
        # inference time) matches what it was trained with.
        self.model.prompts = dict(self.cfg.prompts)
        loss = self._build_loss(self.model)

        # Name fixed to "dev" (not "eval") so the metric key HF's Trainer
        # tracks ("eval_dev_cosine_ap") doesn't collide with its own "eval_"
        # logging prefix.
        eval_metric_name = "dev_cosine_ap"
        evaluator = None
        if eval_data is not None:
            from sentence_transformers.evaluation import BinaryClassificationEvaluator

            eval_query, eval_match, eval_labels = [], [], []
            for query_text, source_text, label in eval_data:
                eval_query.append(query_text)
                eval_match.append(source_text)
                eval_labels.append(0 if str(label) == self.cfg.negative_label else 1)
            evaluator = BinaryClassificationEvaluator(
                sentences1=eval_query,
                sentences2=eval_match,
                labels=eval_labels,
                name="dev",
                similarity_fn_names=["cosine"],
            )

        checkpoint_kwargs: Dict[str, Any] = {}
        callbacks: list[Any] = []
        if evaluator is not None:
            checkpoint_kwargs["eval_strategy"] = "epoch"
        if self.cfg.select_best_checkpoint:
            checkpoint_kwargs.update(
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model=eval_metric_name,
                greater_is_better=True,
                save_total_limit=2,
            )
            if self.cfg.early_stopping_patience is not None:
                from transformers import EarlyStoppingCallback

                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=self.cfg.early_stopping_patience)
                )

        args = SentenceTransformerTrainingArguments(
            output_dir=str(self.cfg.output_dir / "_trainer_output"),
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            prompts=dict(self.cfg.prompts),
            seed=self.cfg.seed,
            report_to=[],
            # HF's Trainer auto-detects CUDA/MPS regardless of the model's own
            # device unless explicitly told not to; honor `cfg.device="cpu"`.
            use_cpu=(self.cfg.device == "cpu"),
            **checkpoint_kwargs,
            **kwargs,
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
            callbacks=callbacks or None,
        )
        trainer.train()
        self.model = trainer.model
        return self.model

    def save(self, **kwargs: Any) -> Path:
        """Persist the fine-tuned SentenceTransformer directory.

        Re-asserts ``model.prompts`` from config as a defensive measure (in
        case anything reset it after ``fit()``), then saves — unlike the
        classification trainer, no id/label bookkeeping is needed here.
        """
        if self.model is None:
            raise ValueError("No trained model available. Call fit() first.")

        self.model.prompts = dict(self.cfg.prompts)
        output_path = resolve_model_output_path(self.cfg.output_dir, self.cfg.output_filename)
        self.model.save(str(output_path))
        return output_path

    def load_artifacts(self, path: str | Path) -> Any:
        """Load a previously saved SentenceTransformer directory."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(path), device=self.cfg.device)
        return self.model
