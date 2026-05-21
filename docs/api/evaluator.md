# Evaluator Module

Tools for assessing detection quality.

## IntertextEvaluator

Evaluate detection results against ground truth annotations.

Computes precision, recall, F1, and other metrics for intertextual link detection.

Binary evaluation remains threshold-based on `judgment_score`.  For multiclass
classifier outputs, call `evaluate_multiclass()` to obtain a per-label
breakdown, for example for `cit` and `cf` labels:

```python
breakdown = evaluator.evaluate_multiclass(
  labels=["cit", "cf"],
  strategy="argmax",
)
```

Ground-truth labels may be numeric (`0`, `1`, `2`) or textual (`cit.`, `cf.`).
By default, `0`/`LABEL_0` map to `no_match`, `1`/`LABEL_1`/`cit.` map to
`cit`, and `2`/`LABEL_2`/`cf.` map to `cf`.  Pass `label_map` to override this
for custom class names.

::: locisimiles.evaluator.IntertextEvaluator
    options:
      heading_level: 3
