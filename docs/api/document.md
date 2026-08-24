# Document Module

Classes for representing and loading text collections.

## TextSegment

An individual unit of text with an identifier.

::: locisimiles.document.TextSegment
    options:
      heading_level: 3

## Document

A collection of text segments with loading utilities.

::: locisimiles.document.Document
    options:
      heading_level: 3

## GroundTruthEntry

One labeled relationship between a query segment and a source segment.

::: locisimiles.ground_truth.GroundTruthEntry
    options:
      heading_level: 3

## GroundTruth

Purpose-built counterpart to `Document`: where a `Document` is a collection of
`TextSegment`s (one corpus), `GroundTruth` is a collection of
`GroundTruthEntry` rows — labeled relationships *between* two corpora,
referenced by segment id rather than duplicating text. Used by
[`IntertextEvaluator`](evaluator.md) and by the [training module](training.md),
which also adds sampling methods for building one via `TrainingData`.

```python
from locisimiles.ground_truth import GroundTruth

gt = GroundTruth("labels.csv")
gt = GroundTruth([{"query_id": "q1", "source_id": "s1", "label": "cit"}])

# Concatenate positives with sampled negatives
combined = positives + negatives
```

::: locisimiles.ground_truth.GroundTruth
    options:
      heading_level: 3
