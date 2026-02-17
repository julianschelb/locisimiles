# Examples

This section provides working examples demonstrating LociSimiles usage.

## Sample Data

The examples use sample Latin texts:

- **Hieronymus samples** - Query texts from Jerome's writings
- **Vergil samples** - Source texts from Virgil's works
- **Ground truth** - Annotated intertextual links for evaluation

## Quick Start Example

```python
--8<-- "examples/example.py"
```

## Jupyter Notebook

For an interactive walkthrough, see the [example notebook](https://github.com/julianschelb/locisimiles/blob/main/examples/example.ipynb).

The notebook covers:

1. **Loading Documents** - Creating Document objects from CSV files
2. **Two-Stage Pipeline** - Using retrieval + classification
3. **Finding Optimal Threshold** - Automatic threshold tuning
4. **Evaluating Different K Values** - Comparing top-k settings
5. **Classification-Only Pipeline** - Exhaustive pairwise comparison

## Two-Stage Pipeline

The recommended approach combines fast retrieval with accurate classification:

```python
from locisimiles.pipeline import ClassificationPipelineWithCandidategeneration
from locisimiles.document import Document

# Load documents
query_doc = Document("./hieronymus_samples.csv", author="Hieronymus")
source_doc = Document("./vergil_samples.csv", author="Vergil")

# Initialize pipeline with pre-trained models
pipeline = ClassificationPipelineWithCandidategeneration(
    classification_name="julian-schelb/PhilBerta-class-latin-intertext-v1",
    embedding_model_name="julian-schelb/SPhilBerta-emb-lat-intertext-v1",
    device="cpu",  # or "cuda", "mps"
)

# Run the pipeline
results = pipeline.run(
    query=query_doc,
    source=source_doc,
    top_k=10  # Number of candidates per query
)
```

## Evaluation

Evaluate your results against ground truth annotations:

```python
from locisimiles.evaluator import IntertextEvaluator

evaluator = IntertextEvaluator(
    query_doc=query_doc,
    source_doc=source_doc,
    ground_truth_csv="./ground_truth.csv",
    pipeline=pipeline,
    top_k=10,
    threshold=0.5,
)

# Evaluate a single query
print(evaluator.evaluate_single_query("hier. adv. iovin. 1.41"))

# Get metrics for all queries
print(evaluator.evaluate(average="macro"))
print(evaluator.evaluate(average="micro"))
```

## Finding the Best Threshold

Automatically find the optimal probability threshold:

```python
best_result, all_thresholds_df = evaluator.find_best_threshold(
    metric="f1",       # Optimize for F1 (or 'precision', 'recall', 'smr')
    average="micro",   # Use micro-averaging
)

print(f"Best threshold: {best_result['best_threshold']}")
print(f"Best F1 score: {best_result['best_f1']:.4f}")
```

## Classification-Only Pipeline

For smaller datasets, use exhaustive pairwise comparison:

```python
from locisimiles.pipeline import ClassificationPipeline

pipeline_clf = ClassificationPipeline(
    classification_name="julian-schelb/PhilBerta-class-latin-intertext-v1",
    device="cpu",
)

results = pipeline_clf.run(
    query=query_doc,
    source=source_doc,
    batch_size=32,
)

# Filter high-probability matches
threshold = 0.7
for query_id, judgments in results.items():
    high_prob = [j for j in judgments if j.judgment_score > threshold]
    if high_prob:
        print(f"Query {query_id}:")
        for j in high_prob:
            print(f"  {j.segment.id}: P={j.judgment_score:.3f}")
```

## Running the Examples

To run the examples locally:

```bash
cd examples
pip install -r requirements.txt
python example.py
```

Or open `example.ipynb` in Jupyter for the interactive version.
