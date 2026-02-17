# Judges

Judges score or classify candidates produced by a generator.

All judges inherit from `JudgeBase` and implement a `judge()` method
returning `CandidateJudgeOutput`.

## JudgeBase

::: locisimiles.pipeline.judge._base.JudgeBase
    options:
      heading_level: 3

## ClassificationJudge

Judge candidates using a transformer sequence-classification model.

::: locisimiles.pipeline.judge.classification.ClassificationJudge
    options:
      heading_level: 3

## ThresholdJudge

Binary decisions based on candidate scores (top-k or threshold).

::: locisimiles.pipeline.judge.threshold.ThresholdJudge
    options:
      heading_level: 3

## IdentityJudge

Pass-through judge that marks every candidate as positive.

::: locisimiles.pipeline.judge.identity.IdentityJudge
    options:
      heading_level: 3
