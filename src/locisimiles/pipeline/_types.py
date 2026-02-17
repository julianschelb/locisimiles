# pipeline/_types.py
"""
Shared type definitions and utilities for pipeline modules.

This module defines the common data structures used across all pipeline
implementations for representing detection results.

Pipeline Architecture
---------------------
Every pipeline follows a two-phase pattern:

1. **Candidate Generation** — narrows the search space, producing a
   ``CandidateGeneratorOutput`` (mapping of query IDs → ``Candidate`` lists).
2. **Judgment** — scores or classifies candidate pairs, producing a
   ``CandidateJudgeOutput`` (mapping of query IDs → ``CandidateJudge`` lists).

The dataclasses ``Candidate`` and ``CandidateJudge`` replace the previous
unnamed tuple types (``SimPair`` / ``FullPair``), giving each field a clear
name.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from locisimiles.document import TextSegment


# ============== DATACLASSES ==============

@dataclass
class Candidate:
    """A single candidate match produced by a candidate-generation stage.

    Attributes:
        segment: The matching source segment.
        score: Relevance score (e.g. cosine similarity, shared-word ratio).
    """
    segment: TextSegment
    score: float


@dataclass
class CandidateJudge:
    """A single scored candidate after the judgment (classification / filtering) stage.

    Attributes:
        segment: The matching source segment.
        candidate_score: Score from candidate generation (``None`` when the
            generator is exhaustive, i.e. all pairs are candidates).
        judgment_score: Final judgment value — e.g. a classification
            probability, a binary 1.0/0.0 decision, or a rule-based score.
    """
    segment: TextSegment
    candidate_score: Optional[float]
    judgment_score: float


# ============== TYPE ALIASES — new names ==============

CandidateGeneratorOutput = Dict[str, List[Candidate]]
"""Mapping from query segment IDs → ranked lists of ``Candidate`` objects.

This is the output type of every candidate-generation stage.
"""

CandidateJudgeOutput = Dict[str, List[CandidateJudge]]
"""Mapping from query segment IDs → lists of ``CandidateJudge`` objects.

This is the standard output type of every pipeline's ``run()`` method
and is consumed by the evaluator.
"""

# Alias so both names work (judge input == candidate-generator output)
CandidateJudgeInput = CandidateGeneratorOutput
"""Alias: the judge receives exactly what the generator produced."""


# ============== BACKWARD-COMPATIBLE ALIASES (deprecated) ==============

Judgment = CandidateJudge
"""*Deprecated* — use ``CandidateJudge`` instead."""

JudgeOutput = CandidateJudgeOutput
"""*Deprecated* — use ``CandidateJudgeOutput`` instead."""

JudgeInput = CandidateJudgeInput
"""*Deprecated* — use ``CandidateJudgeInput`` instead."""

ScoreT = float
"""*Deprecated* — use plain ``float`` instead."""

SimPair = Tuple[TextSegment, float]
"""*Deprecated* — use ``Candidate`` instead."""

FullPair = Tuple[TextSegment, float, float]
"""*Deprecated* — use ``CandidateJudge`` instead."""

SimDict = Dict[str, List[SimPair]]
"""*Deprecated* — use ``CandidateGeneratorOutput`` instead."""

FullDict = Dict[str, List[FullPair]]
"""*Deprecated* — kept for type-checker backward compatibility."""


# ============== UTILITY HELPERS ==============

def pretty_print(results: CandidateJudgeOutput) -> None:
    """
    Print pipeline results in a human-readable format.

    Displays each query segment and its candidate matches with candidate
    scores and judgment scores.

    Args:
        results: Pipeline output in ``CandidateJudgeOutput`` format.

    Example:
        ```python
        from locisimiles.pipeline import pretty_print

        results = pipeline.run(query=query_doc, source=source_doc, top_k=5)
        pretty_print(results)

        # Output:
        # ▶ Query segment 'hier. adv. iovin. 1.41':
        #   verg. aen. 1.1              candidate=+0.823  judgment=0.951
        #   verg. aen. 2.45             candidate=+0.654  judgment=0.234
        ```
    """
    for qid, lst in results.items():
        print(f"\n▶ Query segment {qid!r}:")
        for item in lst:
            if isinstance(item, CandidateJudge):
                seg = item.segment
                cand = item.candidate_score
                judg = item.judgment_score
            else:
                # Backward compat: tuple (segment, sim, prob)
                seg, cand, judg = item  # type: ignore[misc]
            cand_str = f"{cand:+.3f}" if cand is not None else "N/A"
            print(f"  {seg.id:<25}  candidate={cand_str}  judgment={judg:.3f}")


def _unpack_item(item: Any) -> tuple[TextSegment, Optional[float], float]:
    """Extract segment, candidate_score, judgment_score from a result item."""
    if isinstance(item, CandidateJudge):
        return item.segment, item.candidate_score, item.judgment_score
    # Backward compat: tuple (segment, sim, prob)
    seg, cand, judg = item
    return seg, cand, judg


def results_to_csv(
    results: CandidateJudgeOutput,
    path: Union[str, Path],
) -> None:
    """Save pipeline results to a CSV file.

    Writes one row per query-source match with the following columns:

    - ``query_id`` - identifier of the query segment.
    - ``source_id`` - identifier of the matching source segment.
    - ``source_text`` - raw text of the source segment.
    - ``candidate_score`` - score from the candidate-generation stage
      (empty when not available).
    - ``judgment_score`` - final judgment / classification score.

    Args:
        results: Pipeline output in ``CandidateJudgeOutput`` format.
        path: Destination file path (e.g. ``"results.csv"``).

    Example:
        ```python
        from locisimiles.pipeline import results_to_csv

        results = pipeline.run(query=query_doc, source=source_doc, top_k=5)
        results_to_csv(results, "results.csv")
        ```
    """
    path = Path(path)
    fieldnames = [
        "query_id",
        "source_id",
        "source_text",
        "candidate_score",
        "judgment_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid, lst in results.items():
            for item in lst:
                seg, cand, judg = _unpack_item(item)
                writer.writerow({
                    "query_id": qid,
                    "source_id": seg.id,
                    "source_text": seg.text,
                    "candidate_score": cand if cand is not None else "",
                    "judgment_score": judg,
                })


def results_to_json(
    results: CandidateJudgeOutput,
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> None:
    """Save pipeline results to a JSON file.

    Produces a JSON object keyed by query segment ID.  Each value is a
    list of match objects with ``source_id``, ``source_text``,
    ``candidate_score``, and ``judgment_score``.

    Args:
        results: Pipeline output in ``CandidateJudgeOutput`` format.
        path: Destination file path (e.g. ``"results.json"``).
        indent: JSON indentation level (default ``2``).

    Example:
        ```python
        from locisimiles.pipeline import results_to_json

        results = pipeline.run(query=query_doc, source=source_doc, top_k=5)
        results_to_json(results, "results.json")
        ```
    """
    path = Path(path)
    data: Dict[str, list] = {}
    for qid, lst in results.items():
        matches = []
        for item in lst:
            seg, cand, judg = _unpack_item(item)
            matches.append({
                "source_id": str(seg.id),
                "source_text": seg.text,
                "candidate_score": cand,
                "judgment_score": judg,
            })
        data[qid] = matches
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
