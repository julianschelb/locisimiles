# API Reference

This section provides detailed documentation for the LociSimiles Python API, auto-generated from source code docstrings.

## Core Modules

### Document Module

The [Document](document.md) module provides classes for representing and loading text collections:

- `TextSegment` - Individual text unit with ID and content
- `Document` - Container for text segments

### Pipeline Module

The [Pipelines](pipelines.md) module provides the main processing pipelines:

- `RetrievalPipeline` - Semantic similarity retrieval
- `ClassificationPipeline` - Text pair classification
- `ClassificationPipelineWithCandidategeneration` - Two-stage retrieval + classification

### Evaluator Module

The [Evaluator](evaluator.md) module provides tools for assessing detection quality:

- `IntertextEvaluator` - Main evaluation class

## Quick Reference

### Loading Documents

```python
from locisimiles import Document

# From CSV file
doc = Document.from_csv("texts.csv")

# From list of d# From list of d# From list of dre#ords([
    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"  om     {"id"    mpor    {"id"    {"id"pel    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"id"    {"ipip    {"id"    {"id"    {"id"    {"id"    {"id" 0.5)
```

### Evaluating Results

```python
from locisimiles import IntertextEvaluator

evaluator = IntertextEvaluator(predictions, ground_truth)
metrics = evaluator.evaluate()
```
