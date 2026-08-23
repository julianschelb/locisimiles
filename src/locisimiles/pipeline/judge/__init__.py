# pipeline/judge/__init__.py
"""
Judge components that score or classify candidates.

Judges receive the output of a candidate generator and produce scored
results consumed by the evaluator.

Available judges:

- ``ClassificationJudge`` — transformer-based sequence classification
- ``LexicalClassifierJudge`` — trained LogReg/GBDT lexical classification
- ``ThresholdJudge`` — binary decisions from candidate scores
- ``IdentityJudge`` — pass-through (``judgment_score = 1.0``)
"""

from locisimiles.pipeline.judge._base import CandidateJudgeBase, JudgeBase
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.judge.identity import IdentityJudge
from locisimiles.pipeline.judge.lexical_classifier import LexicalClassifierJudge
from locisimiles.pipeline.judge.threshold import ThresholdJudge

__all__ = [
    "CandidateJudgeBase",
    "JudgeBase",  # backward-compat alias
    "ClassificationJudge",
    "LexicalClassifierJudge",
    "ThresholdJudge",
    "IdentityJudge",
]
