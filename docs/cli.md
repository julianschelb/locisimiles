# CLI Reference

LociSimiles provides a single CLI entrypoint:

```bash
locisimiles QUERY.csv SOURCE.csv -o RESULTS.csv [OPTIONS]
```

## Installation

```bash
pip install locisimiles
```

For Word2Vec retrieval, also install:

```bash
pip install "locisimiles[word2vec]"
```

## Arguments

| Argument | Description |
|----------|-------------|
| `QUERY.csv` | Path to query CSV file (`seg_id`, `text`) |
| `SOURCE.csv` | Path to source CSV file (`seg_id`, `text`) |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | required | Output CSV path |
| `--pipeline` | `two-stage` | Pipeline type: `two-stage` or `word2vec-retrieval` |
| `--classification-model` | `julian-schelb/xlm-roberta-large-class-lat-intertext-v1` | Classifier model (two-stage only) |
| `--embedding-model` | `julian-schelb/multilingual-e5-large-emb-lat-intertext-v1` | Embedding model (two-stage only) |
| `--word2vec-model-path` | package default path | Local gensim `.model` path (Word2Vec pipeline) |
| `--word2vec-interval` | `0` | Max token gap for Word2Vec bigrams |
| `--word2vec-order-free` | `False` | Use order-insensitive bigrams |
| `-k, --top-k` | `10` | Number of retrieved candidates per query |
| `-t, --threshold` | `0.85` | Threshold for `above_threshold` in output |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `-v, --verbose` | `False` | Verbose logs |

## Two-Stage Flow

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline two-stage \
    --classification-model julian-schelb/xlm-roberta-large-class-lat-intertext-v1 \
    --embedding-model julian-schelb/multilingual-e5-large-emb-lat-intertext-v1 \
    --top-k 20 \
    --threshold 0.85
```

## Word2Vec Flow

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline word2vec-retrieval \
    --word2vec-model-path ./models/latin_w2v_bamman_lemma300_100_1.model \
    --word2vec-interval 2 \
    --word2vec-order-free \
    --top-k 20 \
    --threshold 0.85
```

If `--word2vec-model-path` is not set, the CLI uses this local default path:

`models/latin_w2v_bamman_lemma300_100_1.model`

The file must exist on disk. No automatic download is performed.

Word2Vec mode expects pre-lemmatized text in the CSV `text` column.

## Output Format

The CLI writes the following columns:

| Column | Description |
|--------|-------------|
| `query_id` | Query segment identifier |
| `query_text` | Query segment text |
| `source_id` | Source segment identifier |
| `source_text` | Source segment text |
| `similarity` | Candidate similarity score |
| `probability` | Final stage score (classification or thresholded retrieval score) |
| `above_threshold` | `Yes` if score >= threshold, else `No` |

## GUI Equivalent

The same Word2Vec settings are available in the GUI under:

1. Pipeline Configuration
2. Pipeline Type: Word2Vec Retrieval (Burns-Style)
3. Word2Vec Model Path / Bigram Interval / Order-Free Bigrams
