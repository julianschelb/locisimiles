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

For TF-IDF, BM25, or the lexical classifier, also install:

```bash
pip install "locisimiles[lexical]"
```

CLTK — used for Latin tokenization/lemmatization — has no release that
supports Python 3.13: the newest 1.x requires `<3.13`, and its 2.x rewrite
requires `>=3.13` with an incompatible API. These three pipelines are
therefore only available on Python 3.10–3.12; on 3.13 the extra installs
without CLTK, and using them raises a clear `ImportError`.

CLTK also needs its Latin corpus data fetched once (a one-time download, not run automatically):

```bash
python -c "from cltk.data.fetch import FetchCorpus; FetchCorpus(language='lat').import_corpus('lat_models_cltk')"
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
| `--pipeline` | `two-stage` | Pipeline type: `two-stage`, `word2vec-retrieval`, `latin-bert-retrieval`, `latin-bert-two-stage`, `tfidf-retrieval`, `bm25-retrieval`, `bm25-two-stage`, or `bm25-lexical-two-stage` |
| `--classification-model` | `julian-schelb/xlm-roberta-large-class-lat-intertext-v1` | Classifier model (two-stage pipelines only) |
| `--embedding-model` | `julian-schelb/multilingual-e5-large-emb-lat-intertext-v1` | Embedding model (two-stage only) |
| `--latin-bert-model` | `xlm-roberta-base` | HuggingFace model for contextual Latin BERT retrieval |
| `--latin-bert-model-path` | none | Optional local model directory for Latin BERT |
| `--latin-bert-max-length` | `256` | Max tokenized sequence length for contextual retrieval |
| `--latin-bert-min-token-length` | `2` | Min token length for contextual scoring |
| `--latin-bert-disable-stopword-filter` | `False` | Disable built-in Latin stopword filtering |
| `--word2vec-model-path` | package default path | Local gensim `.model` path (Word2Vec pipeline) |
| `--word2vec-interval` | `0` | Max token gap for Word2Vec bigrams |
| `--word2vec-order-free` | `False` | Use order-insensitive bigrams |
| `--lexical-disable-lemmatize` | `False` | Disable CLTK lemmatization for TF-IDF/BM25/lexical-classifier pipelines |
| `--tfidf-ngram-max` | `1` | Maximum lemma n-gram size for TF-IDF (1 = unigrams, 2 = unigrams+bigrams) |
| `--bm25-k1` | `1.5` | BM25 term-frequency saturation parameter |
| `--bm25-b` | `0.75` | BM25 length-normalization parameter |
| `--lexical-classifier-path` | none | Path to a `.joblib` artifact from `LexicalClassifierTrainer` (required for `bm25-lexical-two-stage`) |
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

## Latin BERT Retrieval Flow

Token-level contextual similarity using a BERT model (Gong-style approach):

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline latin-bert-retrieval \
    --latin-bert-model ashleygong03/bamman-burns-latin-bert \
    --latin-bert-max-length 256 \
    --top-k 20 \
    --threshold 0.85
```

Or use a local model directory:

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline latin-bert-retrieval \
    --latin-bert-model-path ./models/latinbert \
    --top-k 20
```

## Latin BERT Two-Stage Flow

Combines contextual token retrieval with classification:

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline latin-bert-two-stage \
    --latin-bert-model ashleygong03/bamman-burns-latin-bert \
    --classification-model julian-schelb/xlm-roberta-large-class-lat-intertext-v1 \
    --top-k 20 \
    --threshold 0.85
```

## TF-IDF Retrieval Flow

Lexical retrieval over (optionally lemmatized) Latin text, no trained model required:

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline tfidf-retrieval \
    --tfidf-ngram-max 1 \
    --top-k 20 \
    --threshold 0.85
```

Pass `--lexical-disable-lemmatize` to skip CLTK lemmatization (raw tokens instead).

## BM25 Retrieval Flow

Okapi BM25 over (optionally lemmatized) Latin text — the benchmark's best
single retriever:

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline bm25-retrieval \
    --bm25-k1 1.5 \
    --bm25-b 0.75 \
    --top-k 20 \
    --threshold 0.85
```

## BM25 Two-Stage Flow

BM25 retrieval + cross-encoder classification — the benchmark's "best combined" configuration:

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline bm25-two-stage \
    --classification-model julian-schelb/xlm-roberta-large-class-lat-intertext-v1 \
    --top-k 20 \
    --threshold 0.85
```

## BM25 Lexical Two-Stage Flow

BM25 retrieval + a trained LogReg/GBDT classifier — the benchmark's "best
non-neural" configuration (no neural model required end to end):

```bash
locisimiles query.csv source.csv -o results.csv \
    --pipeline bm25-lexical-two-stage \
    --lexical-classifier-path ./models/lexical_classifier.joblib \
    --top-k 20 \
    --threshold 0.85
```

`--lexical-classifier-path` must point to a `.joblib` artifact produced by
`LexicalClassifierTrainer` (see the [training example](examples.md)).

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

When a multiclass classifier (`bm25-two-stage`, `two-stage`, or
`bm25-lexical-two-stage` with a multiclass artifact) returns class metadata,
the CLI also writes `predicted_class_id`, `predicted_label`, and
`class_probabilities`.

## GUI Equivalent

The same Word2Vec settings are available in the GUI under:

1. Pipeline Configuration
2. Pipeline Type: Word2Vec Retrieval (Burns-Style)
3. Word2Vec Model Path / Bigram Interval / Order-Free Bigrams
