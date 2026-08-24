# API Reference

This section provides detailed documentation for the LociSimiles Python API, auto-generated from source code docstrings.

## Core Modules

### Document Module

The [Document](document.md) module provides classes for representing and loading text collections:

- `TextSegment` - Individual text unit with ID and content
- `Document` - Container for text segments
- `GroundTruthEntry` - One labeled query/source relationship
- `GroundTruth` - Container for labeled query/source pairs

### Pipeline Module

The [Pipelines](pipelines.md) module provides the main processing pipelines:

- `Pipeline` - Generic composer: combine any generator + judge
- `RetrievalPipeline` - Semantic similarity retrieval
- `ClassificationPipeline` - Text pair classification
- `ClassificationPipelineWithCandidateGeneration` - Two-stage retrieval + classification
- `RuleBasedPipeline` - Lexical matching + linguistic filters
- `Word2VecRetrievalPipeline` - Burns-style Word2Vec bigram retrieval
- `LatinBertRetrievalPipeline` / `LatinBertTwoStagePipeline` - Gong-style contextual BERT retrieval
- `TfidfRetrievalPipeline` / `BM25RetrievalPipeline` - Lexical TF-IDF/BM25 retrieval
- `BM25TwoStagePipeline` - BM25 + classification ("best combined")
- `BM25LexicalTwoStagePipeline` - BM25 + trained lexical classifier ("best non-neural")

### Generators Module

The [Generators](generators.md) module provides candidate-generation components:

- `EmbeddingCandidateGenerator` - Semantic embedding similarity
- `ExhaustiveCandidateGenerator` - All-pairs (no filtering)
- `RuleBasedCandidateGenerator` - Lexical matching + linguistic filters
- `Word2VecCandidateGenerator` - Burns-style Word2Vec bigram similarity
- `LatinBertContextualCandidateGenerator` - Gong-style contextual token similarity
- `TfidfCandidateGenerator` - TF-IDF cosine similarity
- `BM25CandidateGenerator` - Okapi BM25 retrieval

### Judges Module

The [Judges](judges.md) module provides scoring/classification components:

- `ClassificationJudge` - Transformer-based sequence classification
- `LexicalClassifierJudge` - Trained LogReg/GBDT lexical classification (no neural model)
- `ThresholdJudge` - Binary decisions from candidate scores
- `IdentityJudge` - Pass-through (judgment_score = 1.0)

### Evaluator Module

The [Evaluator](evaluator.md) module provides tools for assessing detection quality:

- `IntertextEvaluator` - Main evaluation class

### Training Module

The [Training](training.md) module provides trainers for every trainable
approach in the benchmark:

- `TrainingData` - Bundles a query/source `Document` pair with a `GroundTruth`, with negative-sampling methods
- `LexicalClassifierTrainer` - Trains the LogReg/GBDT lexical classifier
- `Word2VecTrainer` - Trains the Burns-style Word2Vec retrieval model
- `ClassificationTrainer` - Fine-tunes the transformer sequence classifier, plus threshold tuning/application and optional best-checkpoint/early-stopping selection
- `EmbeddingTrainer` - Fine-tunes the SentenceTransformer bi-encoder, plus optional best-checkpoint/early-stopping selection
- `cross_validate` - Reproduces the paper's mean±std-across-folds evaluation protocol

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
