# Judges

Judges score or classify candidates produced by a generator.

All judges inherit from `CandidateJudgeBase` and implement a `judge()` method
returning `CandidateJudgeOutput`.

## CandidateJudgeBase

::: locisimiles.pipeline.judge._base.CandidateJudgeBase
    options:
      heading_level: 3

## ClassificationJudge

Judge candidates using a transformer sequence-classification model.

`ClassificationJudge` supports both binary and multiclass sequence classifiers.
For binary classifiers, `judgment_score` remains the positive-class
probability.  For multiclass classifiers, `judgment_score` is the summed
probability of the configured positive classes, while each `CandidateJudge`
also exposes `predicted_class_id`, `predicted_label`, and
`class_probabilities`.

```python
from locisimiles.pipeline.judge import ClassificationJudge

judge = ClassificationJudge(
  classification_name="path-or-hf-id-for-trained-multiclass-model",
  label_names=["no_match", "cit", "cf"],
  positive_labels=["cit", "cf"],
  device="cpu",
)
```

::: locisimiles.pipeline.judge.classification.ClassificationJudge
    options:
      heading_level: 3

## LexicalClassifierJudge

Judge candidates using a trained LogReg/GBDT classifier over TF-IDF/Jaccard/
overlap features (no neural model required). Loads a `.joblib` artifact
produced by `LexicalClassifierTrainer`, and follows the same binary/
multiclass rules as `ClassificationJudge`: `judgment_score` is the
positive-class probability for a binary model, or the summed probability of
the configured positive classes for a multiclass model, which also exposes
`predicted_class_id`, `predicted_label`, and `class_probabilities`.

```python
from locisimiles.pipeline.judge import LexicalClassifierJudge

judge = LexicalClassifierJudge(
  artifact_path="./models/lexical_classifier.joblib",
  positive_labels=["cit", "cf"],
)
```

::: locisimiles.pipeline.judge.lexical_classifier.LexicalClassifierJudge
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
