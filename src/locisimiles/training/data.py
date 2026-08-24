# training/data.py
"""TrainingData: bundles (query_doc, source_doc, ground_truth) for the trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Union

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth, LabelValue
from locisimiles.training import sampling as _sampling


@dataclass
class TrainingData:
    """Bundles a query document, source document, and ground truth for training.

    The single input all pair/label trainers (``LexicalClassifierTrainer``,
    ``ClassificationTrainer``, ``EmbeddingTrainer``) take. Iterating a
    ``TrainingData`` yields resolved ``(query_text, source_text, label)``
    triples — text is looked up by segment id once here, rather than
    duplicated across each trainer.

    Negative-sampling methods are available directly as chainable, immutable
    methods:

    ```python
    data = TrainingData(query_doc, source_doc, positives).sample_random_negatives(n_per_query=5)
    ```

    Attributes:
        query_doc: Query corpus.
        source_doc: Source corpus.
        ground_truth: Labeled query/source pairs.
    """

    query_doc: Document
    source_doc: Document
    ground_truth: GroundTruth

    def __len__(self) -> int:
        return len(self.ground_truth)

    def __iter__(self) -> Iterator[Tuple[str, str, LabelValue]]:
        for entry in self.ground_truth:
            yield (
                self.query_doc[entry.query_id].text,
                self.source_doc[entry.source_id].text,
                entry.label,
            )

    def __repr__(self) -> str:
        return (
            f"TrainingData(query_segments={len(self.query_doc)}, "
            f"source_segments={len(self.source_doc)}, pairs={len(self)})"
        )

    def __add__(self, other: TrainingData) -> TrainingData:
        if not isinstance(other, TrainingData):
            return NotImplemented
        return TrainingData(self.query_doc, self.source_doc, self.ground_truth + other.ground_truth)

    # ---------- negative sampling ----------

    def sample_random_pairs(
        self, *, n_per_query: int = 1, seed: int = 42, label: LabelValue = "no_match"
    ) -> TrainingData:
        """Add ⟨rnd,rnd⟩ negatives: independently-random (query, source) pairs.

        Included for completeness/parity with the paper's ablation, which
        found this the weakest of the four methods.
        """
        negatives = _sampling.sample_random_pairs(
            query_doc=self.query_doc,
            source_doc=self.source_doc,
            positives=self.ground_truth,
            n_per_query=n_per_query,
            seed=seed,
            label=label,
        )
        return TrainingData(self.query_doc, self.source_doc, self.ground_truth + negatives)

    def sample_random_negatives(
        self, *, n_per_query: int = 1, seed: int = 42, label: LabelValue = "no_match"
    ) -> TrainingData:
        """Add ⟨qry,rnd⟩ negatives — **the default method**.

        For every query segment, samples a random non-positive source
        segment. Needs no model loading, so this is the package's default,
        first-reached-for sampling method.
        """
        negatives = _sampling.sample_random_negatives(
            query_doc=self.query_doc,
            source_doc=self.source_doc,
            positives=self.ground_truth,
            n_per_query=n_per_query,
            seed=seed,
            label=label,
        )
        return TrainingData(self.query_doc, self.source_doc, self.ground_truth + negatives)

    def sample_hard_negatives(
        self,
        *,
        n_per_query: int = 1,
        embedding_model_name: str = _sampling.DEFAULT_HARD_NEGATIVE_MODEL_NAME,
        device: Optional[Union[str, int]] = None,
        label: LabelValue = "no_match",
    ) -> TrainingData:
        """Add ⟨qry,sim⟩ negatives: nearest non-positive neighbors by embedding similarity."""
        negatives = _sampling.sample_hard_negatives(
            query_doc=self.query_doc,
            source_doc=self.source_doc,
            positives=self.ground_truth,
            n_per_query=n_per_query,
            embedding_model_name=embedding_model_name,
            device=device,
            label=label,
        )
        return TrainingData(self.query_doc, self.source_doc, self.ground_truth + negatives)

    def sample_mixed_negatives(
        self,
        *,
        n_random_per_query: int = 5,
        n_hard_per_query: int = 5,
        embedding_model_name: str = _sampling.DEFAULT_HARD_NEGATIVE_MODEL_NAME,
        device: Optional[Union[str, int]] = None,
        seed: int = 42,
        label: LabelValue = "no_match",
    ) -> TrainingData:
        """Add ⟨qry,mix⟩ negatives — the paper's best-performing method on every
        reported metric, combining random and embedding-mined hard negatives
        for the same query. Offered as an explicit opt-in upgrade from the
        default ``sample_random_negatives``.

        Implemented as a thin composition of ``sample_random_negatives`` and
        ``sample_hard_negatives`` — no separate sampling logic.
        """
        return self.sample_random_negatives(
            n_per_query=n_random_per_query, seed=seed, label=label
        ).sample_hard_negatives(
            n_per_query=n_hard_per_query,
            embedding_model_name=embedding_model_name,
            device=device,
            label=label,
        )
