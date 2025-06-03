# Loci Similes

**LociSimiles** is a Python package for finding intertextual links in Latin literature using pre-trained language models.

## Basic Usage

```python

# Load example query and source documents
query_doc = Document("../data/hieronymus_samples.csv")
source_doc = Document("../data/vergil_samples.csv")

# Load the pipeline with pre-trained models
pipeline = ClassificationPipelineWithCandidategeneration(
    classification_name="...",
    embedding_model_name="...",
    device="cpu",
)

# Run the pipeline with the query and source documents
results = pipeline.run(
    query=query_doc,    # Query document
    source=source_doc,  # Source document
    top_k=3             # Number of top similar candidates to classify
)

pretty_print(results)
```
