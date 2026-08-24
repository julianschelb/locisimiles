# training/sampling.py
"""Standalone negative-sampling functions underlying ``TrainingData``'s sampling methods.

Implements all four negative-construction methods studied in the paper's own
ablation (see :class:`~locisimiles.training.data.TrainingData`): random pairs
(⟨rnd,rnd⟩), random negatives (⟨qry,rnd⟩), and hard negatives (⟨qry,sim⟩).
Mixed negatives (⟨qry,mix⟩) is a composition of the latter two and is
implemented directly on ``TrainingData`` rather than here.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Set, Tuple, Union

from locisimiles.document import Document
from locisimiles.ground_truth import GroundTruth, GroundTruthEntry, LabelValue

DEFAULT_HARD_NEGATIVE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _seed_for(base_seed: int, key: str) -> int:
    """Deterministic per-key seed derived from a base seed."""
    digest = hashlib.sha256(f"{base_seed}:{key}".encode()).hexdigest()
    return int(digest[:16], 16)


def _positives_by_query(positives: GroundTruth) -> Dict[object, Set[object]]:
    grouped: Dict[object, Set[object]] = {}
    for entry in positives:
        grouped.setdefault(entry.query_id, set()).add(entry.source_id)
    return grouped


def sample_random_pairs(
    *,
    query_doc: Document,
    source_doc: Document,
    positives: GroundTruth,
    n_per_query: int = 1,
    seed: int = 42,
    label: LabelValue = "no_match",
) -> GroundTruth:
    """⟨rnd,rnd⟩ — independently-random (query, source) pairs.

    Draws ``n_per_query * len(query_doc)`` pairs by picking a query segment
    and a source segment *independently* and uniformly at random (not
    conditioned on any specific query, unlike the other sampling functions
    here), skipping any draw that happens to coincide with a known positive
    in ``positives``. Included for completeness/parity with the paper's
    ablation — its own results table shows this is the weakest of the four
    methods.

    Args:
        query_doc: Query corpus.
        source_doc: Source corpus.
        positives: Known positive pairs to avoid mislabeling as negative.
        n_per_query: Target number of random pairs per query segment.
        seed: RNG seed for reproducibility.
        label: Label assigned to sampled negatives.

    Returns:
        A ``GroundTruth`` of newly sampled negative pairs.
    """
    known_pairs = {(entry.query_id, entry.source_id) for entry in positives}
    query_ids = list(query_doc.ids())
    source_ids = list(source_doc.ids())
    if not query_ids or not source_ids:
        return GroundTruth()

    total = max(0, int(n_per_query)) * len(query_ids)
    rng = random.Random(seed)
    entries: List[GroundTruthEntry] = []
    seen: Set[Tuple[object, object]] = set()
    max_attempts = total * 20 + 100
    attempts = 0
    while len(entries) < total and attempts < max_attempts:
        attempts += 1
        q_id = rng.choice(query_ids)
        s_id = rng.choice(source_ids)
        pair = (q_id, s_id)
        if pair in known_pairs or pair in seen:
            continue
        seen.add(pair)
        entries.append(GroundTruthEntry(query_id=q_id, source_id=s_id, label=label))

    return GroundTruth(entries)


def sample_random_negatives(
    *,
    query_doc: Document,
    source_doc: Document,
    positives: GroundTruth,
    n_per_query: int = 1,
    seed: int = 42,
    label: LabelValue = "no_match",
) -> GroundTruth:
    """⟨qry,rnd⟩ — for every query segment, ``n_per_query`` random non-positive
    source segments.

    Uses a seed derived deterministically per query id (rather than one
    shared RNG stream) so results are reproducible regardless of query
    iteration order.

    Args:
        query_doc: Query corpus.
        source_doc: Source corpus.
        positives: Known positive pairs to exclude from sampling.
        n_per_query: Number of negatives to sample per query segment.
        seed: Base RNG seed for reproducibility.
        label: Label assigned to sampled negatives.

    Returns:
        A ``GroundTruth`` of newly sampled negative pairs.
    """
    positives_by_query = _positives_by_query(positives)
    source_ids = list(source_doc.ids())

    entries: List[GroundTruthEntry] = []
    for query_segment in query_doc:
        q_id = query_segment.id
        excluded = positives_by_query.get(q_id, set())
        candidates = [s_id for s_id in source_ids if s_id not in excluded]
        if not candidates:
            continue
        rng = random.Random(_seed_for(seed, str(q_id)))
        k = min(max(0, int(n_per_query)), len(candidates))
        for s_id in rng.sample(candidates, k):
            entries.append(GroundTruthEntry(query_id=q_id, source_id=s_id, label=label))

    return GroundTruth(entries)


def sample_hard_negatives(
    *,
    query_doc: Document,
    source_doc: Document,
    positives: GroundTruth,
    n_per_query: int = 1,
    embedding_model_name: str = DEFAULT_HARD_NEGATIVE_MODEL_NAME,
    device: Optional[Union[str, int]] = None,
    label: LabelValue = "no_match",
) -> GroundTruth:
    """⟨qry,sim⟩ — for every query segment, the top ``n_per_query``
    *non-positive* source segments ranked by embedding similarity.

    Reuses :class:`~locisimiles.pipeline.generator.embedding.EmbeddingCandidateGenerator`
    to mine negatives, matching the paper's "a pretrained embedding model"
    nearest-neighbor mining. ``embedding_model_name`` should be a
    general-purpose pretrained model, not the fine-tuned model being
    trained — mining hard negatives with the very model about to be
    fine-tuned would be circular.

    Args:
        query_doc: Query corpus.
        source_doc: Source corpus.
        positives: Known positive pairs to exclude from the mined candidates.
        n_per_query: Number of hard negatives to keep per query segment.
        embedding_model_name: Pretrained sentence-transformer used to mine
            negatives.
        device: Torch device string for the mining model.
        label: Label assigned to sampled negatives.

    Returns:
        A ``GroundTruth`` of newly mined negative pairs.
    """
    from locisimiles.pipeline.generator.embedding import EmbeddingCandidateGenerator

    positives_by_query = _positives_by_query(positives)
    n = max(0, int(n_per_query))
    if n == 0 or len(query_doc) == 0 or len(source_doc) == 0:
        return GroundTruth()

    max_positives = max((len(v) for v in positives_by_query.values()), default=0)
    top_k = min(n + max_positives, len(source_doc))
    top_k = max(top_k, 1)

    generator = EmbeddingCandidateGenerator(
        embedding_model_name=embedding_model_name, device=device
    )
    ranked = generator.generate(
        query=query_doc,
        source=source_doc,
        top_k=top_k,
        # A generic pretrained model has no configured "query"/"match"
        # prompts; disable prompt lookups to avoid a ValueError.
        query_prompt_name="",
        source_prompt_name="",
    )

    entries: List[GroundTruthEntry] = []
    for q_id, candidates in ranked.items():
        excluded = positives_by_query.get(q_id, set())
        chosen = [c for c in candidates if c.segment.id not in excluded][:n]
        for candidate in chosen:
            entries.append(
                GroundTruthEntry(query_id=q_id, source_id=candidate.segment.id, label=label)
            )

    return GroundTruth(entries)
