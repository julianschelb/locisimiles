# pipeline/_types.py
"""
Shared type definitions and utilities for pipeline modules.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from locisimiles.document import TextSegment

# ============== TYPE ALIASES ==============

ScoreT = float
SimPair = Tuple[TextSegment, ScoreT]              # (segment, cosine-sim)
FullPair = Tuple[TextSegment, ScoreT, ScoreT]      # (+ prob-positive)
SimDict = Dict[str, List[SimPair]]
FullDict = Dict[str, List[FullPair]]


# ============== UTILITY HELPERS ==============

def pretty_print(results: FullDict) -> None:
    """Human-friendly dump of *run()* output."""
    for qid, lst in results.items():
        print(f"\n▶ Query segment {qid!r}:")
        for src_seg, sim, ppos in lst:
            sim_str = f"{sim:+.3f}" if sim is not None else "N/A"
            print(f"  {src_seg.id:<25}  sim={sim_str}  P(pos)={ppos:.3f}")
