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

LociSimiles provides three main pipeline types:

### 1. Retrieval Pipeline

Uses semantic embeddings to find similar passages:

```python
from locisimiles import RetrievalPipeline

pipeline = RetrievalPipeline(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

results = pipeline.retrieve(source_doc, target_doc, top_k=10)
```

### 2. Classification Pipeline

Uses a transformer model to classify text pairs:

```python
from locisimiles import ClassificationPipeline

pipeline = ClassificationPipeline(
    model_name="bert-base-uncased"
)

results = pipeline.classify(pairs)
```

### 3. Two-Stage Pipeline

Combines retrieval and classification for best results:

```python
from locisimiles import TwoStagePipeline

pipeline = TwoStagePipeline(
    retrieval_model="sentence-transformers/all-MiniLM-L6-v2",
    classification_model="bert-base-uncased"
)

results = pipeline.run(source_doc, target_doc)
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
