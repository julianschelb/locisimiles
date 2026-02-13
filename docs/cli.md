# CLI Reference

LociSimiles provides a command-line interface for common workflows.

## Installation

The CLI is installed automatically with the package:

```bash
pip install locisimiles
```

## Commands

### `locisimiles run`

Run the intertextual detection pipeline on source and target documents.

```bash
locisimiles run SOURCE TARGET [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `SOURCE` | Path to the source CSV file |
| `TARGET` | Path to the target CSV file |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `results.csv` | Output file path |
| `--model`, `-m` | `sentence-transformers/all-MiniLM-L6-v2` | Model name or path |
| `--top-k`, `-k` | `10` | Number of candidates to retrieve |
| `--threshold`, `-t` | `0.5` | Classification threshold |
| `--batch-size`, `-b` | `32` | Batch size for processing |

#### Examples

Basic usage:

```bash
locisimiles run source.csv target.csv
```

With custom output and model:

```bash
locisimiles run source.csv target.csv \
    --output results.csv \
    --model bert-base-multilingual-cased \
    --top-k 20
```

### `locisimiles evaluate`

Evaluate detection results against ground truth.

```bash
locisimiles evaluate PREDICTIONS GROUND_TRUTH [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `PREDICTIONS` | Path to predictions CSV file |
| `GROUND_TRUTH` | Path to ground truth CSV file |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `None` | Output file for metrics (prints to stdout if not specified) |

#### Examples

```bash
locisimiles evaluate results.csv ground_truth.csv
```

Save metrics to file:

```bash
locisimiles evaluate results.csv ground_truth.csv -o metrics.json
```

## Input File Formats

### Source/Target CSV

CSV files should contain at minimum an ID column and a text column:

```csv
id,text
1,"Arma virumque cano Troiae qui primus ab oris"
2,"Italiam fato profugus Laviniaque venit"
```

### Ground Truth CSV

Ground truth files should contain query-reference pairs with labels:

```csv
query_id,reference_id,label
1,42,1
2,15,0
```

Where `label` is `1` for true matches and `0` for non-matches.

## Output Format

The pipeline outputs a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `query_id` | ID of the source text segment |
| `reference_id` | ID of the matched target segment |
| `score` | Similarity/classification score |
| `above_threshold` | Whether the score exceeds the threshold |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LOCISIMILES_CACHE_DIR` | Directory for model caching |
| `LOCISIMILES_DEVICE` | Device for computation (`cpu`, `cuda`, `mps`) |
