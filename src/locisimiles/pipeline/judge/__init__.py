# pipeline/judge/__init__.py
"""
Judge components that score or classify candidates.

Judges receive the output of a candidate generator and produce scored
results consumed by the evaluator.

Available judges:

- ``ClassificationJudge`` — transformer-based sequence classification
- ``ThresholdJudge`` — binary decisions from candidate scores
- ``IdentityJudge`` — pass-through (``judgment_score = 1.0``)
"""
from locisimiles.pipeline.judge._base import JudgeBase
from locisimiles.pipeline.judge.classification import ClassificationJudge
from locisimiles.pipeline.judge.threshold import ThresholdJudge
from locisimiles.pipeline.judge.identity import IdentityJudge

__all__ = [
    "JudgeBase",
    "ClassificationJudge",
    "ThresholdJudge",
    "IdentityJudge",
]
