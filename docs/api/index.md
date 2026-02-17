# API Reference

This section provides detailed documentation for the LociSimiles Python API, auto-generated from source code docstrings.

## Core Modules

### Document Module

The [Document](document.md) module provides classes for representing and loading text collections:

- `TextSegment` - Individual text unit with ID and content
- `Document` - Container for text segments

### Pipeline Module

The [Pipelines](pipelines.md) module provides the main processing pipelines:

- `Pipeline` - Generic composer: combine any generator + judge
- `RetrievalPipeline` - Semantic similarity retrieval
- `ClassificationPipeline` - Text pair classification
- `ClassificationPipelineWithCandidategeneration` - Two-stage retrieval + classification
- `RuleBasedPipeline` - Lexical matching + linguistic filters

### Generators Module

The [Generators](generators.md) module provides candidate-generation components:

- `EmbeddingCandidateGenerator` - Semantic embedding similarity
- `ExhaustiveCandidateGenerator` - All-pairs (no filtering)
- `RuleBasedCandidateGenerator` - Lexical matching + linguistic filters

### Judges Module

The [Judges](judges.md) module provides scoring/classification components:

- `ClassificationJudge` - Transformer-based sequence classification
- `ThresholdJudge` - Binary decisions from candidate scores
- `IdentityJudge` - Pass-through (judgment_score = 1.0)

### Evaluator Module

The [Evaluator](evaluator.md) module provides tools for assessing detection quality:

- `IntertextEvaluator` - Main evaluation class

## Quick Reference

### Loading Documents

```python
from locisimiles import Document

doc = Document("texts.csv")
```

### Saving Results

```python
# Save from a pipeline instance
results = pipeline.run(query=query_doc, source=source_doc, top_k=10)
pipeline.to_csv("results.csv")
pipeline.to_json("results.json")

# Or use standalone functions
from locisimiles.pipeline import results_to_csv, results_to_json
results_to_csv(results, "results.csv")
results_to_json(results, "results.json")
```

### Evaluating Results

```python
from locisimiles import IntertextEvaluator

evaluator = IntertextEvaluator(predictions, ground_truth)
metrics = evaluator.evaluate()
```
