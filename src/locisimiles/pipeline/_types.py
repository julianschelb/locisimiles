# pipeline/_types.py
"""
Shared type definitions and utilities for pipeline modules.

This module defines the common data structures used across all pipeline
implementations for representing detection results.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from locisimiles.document import TextSegment

# ============== TYPE ALIASES ==============

ScoreT = float
"""Type alias for similarity or probability scores (float between 0 and 1)."""

SimPair = Tuple[TextSegment, ScoreT]
"""
A tuple of (TextSegment, similarity_score) representing a candidate match.

Used in retrieval-only pipelines where only similarity scores are computed.
"""

FullPair = Tuple[TextSegment, ScoreT, ScoreT]
"""
A tuple of (TextSegment, similarity_score, probability) for classified matches.

The three elements are:
    - TextSegment: The matching source segment
    - similarity_score: Cosine similarity from retrieval (may be None)
    - probability: Classification probability P(positive)
"""

SimDict = Dict[str, List[SimPair]]
"""
Mapping from query segment IDs to lists of (segment, similarity) pairs.

Used as intermediate result format in retrieval pipelines.
"""

FullDict = Dict[str, List[FullPair]]
"""
Mapping from query segment IDs to lists of (segment, similarity, probability) tuples.

This is the standard output format for all pipelines, used by the evaluator.

Example:
    ```python
    results: FullDict = {
        "hier. adv. iovin. 1.41": [
            (TextSegment(...), 0.82, 0.95),  # High confidence match
            (TextSegment(...), 0.65, 0.23),  # Low confidence
        ],
        "hier. adv. iovin. 1.42": [...],
    }
    ```
"""


# ============== UTILITY HELPERS ==============

def pretty_print(results: FullDict) -> None:
    """
    Print pipeline results in a human-readable format.
    
    Displays each query segment and its candidate matches with similarity
    scores and classification probabilities.
    
    Args:
        results: Pipeline output in FullDict format.
    
    Example:
        ```python
        from locisimiles.pipeline import pretty_print
        
        results = pipeline.run(query=query_doc, source=source_doc, top_k=5)
        pretty_print(results)
        
        # Output:
        # ▶ Query segment 'hier. adv. iovin. 1.41':
        #   verg. aen. 1.1              sim=+0.823  P(pos)=0.951
        #   verg. aen. 2.45             sim=+0.654  P(pos)=0.234
        ```
    """
    for qid, lst in results.items():
        print(f"\n▶ Query segment {qid!r}:")
        for src_seg, sim, ppos in lst:
            sim_str = f"{sim:+.3f}" if sim is not None else "N/A"
            print(f"  {src_seg.id:<25}  sim={sim_str}  P(pos)={ppos:.3f}")
