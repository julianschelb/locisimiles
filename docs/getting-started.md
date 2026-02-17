# Getting Started

This guide will help you get started with LociSimiles for finding intertextual links in Latin literature.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installing from PyPI

```bash
pip install locisimiles
```

### Installing from Source

```bash
git clone https://github.com/julianschelb/locisimiles.git
cd locisimiles
pip install -e ".[dev]"
```

## Basic Concepts

### Documents and Segments

LociSimiles works with **Documents** containing **TextSegments**. Each segment represents a unit of text (e.g., a verse, sentence, or passage).

```python
from locisimiles import Document, TextSegment

# Create segments manually
segments = [
    TextSegment(id="1", text="Arma virumque cano"),
    TextSegment(id="2", text="Troiae qui primus ab oris"),
]

# Create a document
doc = Document(segments=segments)
```

### Loading from CSV

Documents are typically loaded from CSV files:

```python
doc = Document.from_csv("texts.csv")
```

The CSV should have columns for `id` and `text` (column names are configurable).

## Pipelines

LociSimiles provides a **modular pipeline architecture** where you compose
a **generator** (candidate selection) with a **judge** (scoring/classification)
using the generic `Pipeline` class.

### Modular Approach (Recommended)

```python
from locisimiles import Pipeline
from locisimiles.pipeline.generator import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge import ClassificationJudge

pipeline = Pipeline(
    generator=EmbeddingCandidateGenerator(device="cpu"),
    judge=ClassificationJudge(device="cpu"),
)

results = pipeline.run(query=query_doc, source=source_doc, top_k=10)
```

Available **generators**:

| Generator | Description |
|-----------|-------------|
| `EmbeddingCandidateGenerator` | Semantic similarity via sentence transformers + ChromaDB |
| `ExhaustiveCandidateGenerator` | All pairs — no filtering |
| `RuleBasedCandidateGenerator` | Lexical matching + linguistic filters |

Available **judges**:

| Judge | Description |
|-------|-------------|
| `ClassificationJudge` | Transformer sequence classification (P(positive)) |
| `ThresholdJudge` | Binary decisions from candidate scores (top-k or threshold) |
| `IdentityJudge` | Pass-through — `judgment_score = 1.0` |

### Legacy Pipeline Classes

The pre-composed pipeline classes are still available for convenience:

```python
from locisimiles import (
    ClassificationPipelineWithCandidategeneration,  # embedding + classification
    ClassificationPipeline,                         # exhaustive + classification
    RetrievalPipeline,                              # embedding + threshold
    RuleBasedPipeline,                              # rule-based + identity
)
```

## Evaluation

Use the `IntertextEvaluator` to assess detection quality:

```python
from locisimiles import IntertextEvaluator

evaluator = IntertextEvaluator(
    predictions=predictions,
    ground_truth=ground_truth
)

metrics = evaluator.evaluate()
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
```

## Next Steps

- See the [CLI Reference](cli.md) for command-line usage
- Explore the [API Reference](api/index.md) for detailed documentation
- Check out the [examples](https://github.com/julianschelb/locisimiles/tree/main/examples) for complete workflows
