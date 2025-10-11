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

## Command-Line Interface

LociSimiles provides a command-line tool for running the pipeline directly from the terminal:

### Basic Usage

```bash
locisimiles query.csv source.csv -o results.csv
```

### Advanced Usage

```bash
locisimiles query.csv source.csv -o results.csv \
  --classification-model julian-schelb/PhilBerta-class-latin-intertext-v1 \
  --embedding-model julian-schelb/SPhilBerta-emb-lat-intertext-v1 \
  --top-k 20 \
  --threshold 0.7 \
  --device cuda \
  --verbose
```

### Options

- **Input/Output:**
  - `query`: Path to query document CSV file (columns: `seg_id`, `text`)
  - `source`: Path to source document CSV file (columns: `seg_id`, `text`)
  - `-o, --output`: Path to output CSV file for results (required)

- **Models:**
  - `--classification-model`: HuggingFace model for classification (default: PhilBerta-class-latin-intertext-v1)
  - `--embedding-model`: HuggingFace model for embeddings (default: SPhilBerta-emb-lat-intertext-v1)

- **Pipeline Parameters:**
  - `-k, --top-k`: Number of top candidates to retrieve per query segment (default: 10)
  - `-t, --threshold`: Classification probability threshold for filtering results (default: 0.5)

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
