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

# Save results to CSV or JSON
pipeline.to_csv("results.csv")
pipeline.to_json("results.json")
```

## Command-Line Interface

LociSimiles provides a command-line tool for running the pipeline directly from the terminal:

### Basic Usage

```bash
locisimiles query.csv source.csv -o results.csv
```

### Two-Stage Pipeline Example

```bash
locisimiles query.csv source.csv -o results.csv \
  --pipeline two-stage \
  --classification-model julian-schelb/xlm-roberta-large-class-lat-intertext-v1 \
  --embedding-model julian-schelb/multilingual-e5-large-emb-lat-intertext-v1 \
  --top-k 20 \
  --threshold 0.85 \
  --device cuda \
  --verbose
```

### Word2Vec Retrieval Example

```bash
locisimiles query.csv source.csv -o results.csv \
  --pipeline word2vec-retrieval \
  --word2vec-model-path ./models/latin_w2v_bamman_lemma300_100_1.model \
  --word2vec-interval 2 \
  --word2vec-order-free \
  --top-k 20 \
  --threshold 0.85
```

If `--word2vec-model-path` is not provided, the CLI expects a local model at:

`models/latin_w2v_bamman_lemma300_100_1.model`

Word2Vec mode requires pre-lemmatized input in the same CSV format (`seg_id`, `text`).

### Options

- **Input/Output:**
  - `query`: Path to query document CSV file (columns: `seg_id`, `text`)
  - `source`: Path to source document CSV file (columns: `seg_id`, `text`)
  - `-o, --output`: Path to output CSV file for results (required)

- **Models:**
  - `--classification-model`: HuggingFace model for classification (default: xlm-roberta-large-class-lat-intertext-v1)
  - `--embedding-model`: HuggingFace model for embeddings (default: multilingual-e5-large-emb-lat-intertext-v1)
  - `--word2vec-model-path`: Local path to a gensim `.model` file (Word2Vec pipeline)

- **Pipeline Parameters:**
  - `--pipeline`: Select `two-stage` or `word2vec-retrieval` (default: `two-stage`)
  - `-k, --top-k`: Number of top candidates to retrieve per query segment (default: 10)
  - `-t, --threshold`: Decision threshold for output filtering (default: 0.85)
  - `--word2vec-interval`: Max token gap for Word2Vec bigrams (default: 0)
  - `--word2vec-order-free`: Enable order-insensitive Word2Vec bigrams

- **Device:**
  - `--device`: Choose `auto`, `cuda`, `mps`, or `cpu` (default: auto-detect)

- **Other:**
  - `-v, --verbose`: Enable detailed progress output
  - `-h, --help`: Show help message

### Output Format

The CLI saves results to a CSV file with the following columns:
- `query_id`: Query segment identifier
- `query_text`: Query text content
- `source_id`: Source segment identifier
- `source_text`: Source text content
- `similarity`: Cosine similarity score (0-1)
- `probability`: Classification confidence (0-1)
- `above_threshold`: "Yes" if probability ≥ threshold, otherwise "No"


## Optional Gradio GUI

Install the optional GUI extra to experiment with a minimal Gradio front end:

```bash
pip install locisimiles[gui]
```

Launch the interface from the command line:

```bash
locisimiles-gui
```

In the GUI, choose **Word2Vec Retrieval (Burns-Style)** in Pipeline Configuration to enable Word2Vec controls:

- Word2Vec Model Path: local gensim `.model` file
- Bigram Interval: token gap for bigram generation
- Order-Free Bigrams: optional order-insensitive matching

If the model path is invalid or missing, processing fails with a clear error message.
