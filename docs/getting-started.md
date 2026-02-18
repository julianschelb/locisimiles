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

LociSimiles provides ready-to-use pipelines for detecting intertextual links.
Each pipeline takes a **query document** and a **source document** and returns
scored matches.

### Two-Stage Pipeline

The recommended pipeline for most use cases. It first retrieves the most
promising candidates using embedding similarity, then classifies each
candidate pair with a fine-tuned transformer model.

```python
from locisimiles import ClassificationPipelineWithCandidategeneration
from locisimiles import Document

# Load documents
query = Document("query.csv")
source = Document("source.csv")

# Define pipeline
pipeline = ClassificationPipelineWithCandidategeneration(
    classification_name="julian-schelb/PhilBerta-class-latin-intertext-v1",
    embedding_model_name="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
    device="cpu",  # or "cuda", "mps"
)

# Run pipeline
results = pipeline.run(query=query, source=source, top_k=10)
```

### Classification Pipeline

Classifies **every possible** query–source pair using a fine-tuned
sequence-classification model. More thorough but slower — best suited for
smaller datasets.

```python
from locisimiles import ClassificationPipeline
from locisimiles import Document

# Load documents
query = Document("query.csv")
source = Document("source.csv")

# Define pipeline
pipeline = ClassificationPipeline(
    classification_name="julian-schelb/PhilBerta-class-latin-intertext-v1",
    device="cpu",
)

# Run pipeline
results = pipeline.run(query=query, source=source, batch_size=32)
```

### Retrieval Pipeline

A fast, lightweight pipeline that ranks source segments by embedding
similarity and applies a top-k or threshold criterion. No classification
model needed.

```python
from locisimiles import RetrievalPipeline
from locisimiles import Document

# Load documents
query = Document("query.csv")
source = Document("source.csv")

# Define pipeline
pipeline = RetrievalPipeline(
    embedding_model_name="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
    device="cpu",
)

# Run pipeline
results = pipeline.run(query=query, source=source, top_k=5)
```

### Rule-Based Pipeline

A purely lexical pipeline that does not require any neural models.
It finds shared words between segments and applies distance, punctuation,
and optional POS / similarity filters.

```python
from locisimiles import RuleBasedPipeline
from locisimiles import Document

# Load documents
query = Document("query.csv")
source = Document("source.csv")

# Define pipeline
pipeline = RuleBasedPipeline(min_shared_words=2, max_distance=3)

# Run pipeline
results = pipeline.run(query=query, source=source)
```

### Pipeline Summary

| Pipeline | Speed | Models required | Best for |
|----------|-------|-----------------|----------|
| `ClassificationPipelineWithCandidategeneration` | Medium | Embedding + classifier | Most use cases |
| `ClassificationPipeline` | Slow | Classifier | Small datasets, exhaustive comparison |
| `RetrievalPipeline` | Fast | Embedding | Quick similarity search |
| `RuleBasedPipeline` | Fast | None | No GPU, lexical matching |

### Saving Results

Save pipeline output to CSV or JSON:

```python
results = pipeline.run(query=query, source=source, top_k=10)

# Save directly from the pipeline
pipeline.to_csv("results.csv")
pipeline.to_json("results.json")

# Or use standalone utility functions
from locisimiles.pipeline import results_to_csv, results_to_json
results_to_csv(results, "results.csv")
results_to_json(results, "results.json")
```

## Building Custom Pipelines

For advanced use cases you can compose your own pipeline from individual
**generators** and **judges** using the generic `Pipeline` class.

A **generator** selects candidate source segments for each query segment.
A **judge** then scores or classifies each candidate pair.

### Available Generators

| Generator | Description |
|-----------|-------------|
| `EmbeddingCandidateGenerator` | Semantic similarity via sentence transformers + ChromaDB |
| `ExhaustiveCandidateGenerator` | All pairs — no filtering |
| `RuleBasedCandidateGenerator` | Lexical matching + linguistic filters |

### Available Judges

| Judge | Description |
|-------|-------------|
| `ClassificationJudge` | Transformer sequence classification (P(positive)) |
| `ThresholdJudge` | Binary decisions from candidate scores (top-k or threshold) |
| `IdentityJudge` | Pass-through — `judgment_score = 1.0` |

### Example

```python
from locisimiles import Pipeline
from locisimiles.pipeline.generator import EmbeddingCandidateGenerator
from locisimiles.pipeline.judge import ClassificationJudge

pipeline = Pipeline(
    generator=EmbeddingCandidateGenerator(device="cpu"),
    judge=ClassificationJudge(device="cpu"),
)

results = pipeline.run(query=query_doc, source=source_doc, top_k=10)

# You can also run each stage separately
candidates = pipeline.generate_candidates(query=query_doc, source=source_doc, top_k=10)
results = pipeline.judge_candidates(query=query_doc, candidates=candidates)
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
